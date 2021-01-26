import numpy as np
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.distributions import MultivariateNormal
from torch.optim import Adam

from mlp_policy import Policy
from mlp_critic import Value


class PPO:
    def __init__(self, env, conf=dict(), device=torch.device('cpu')):
        self.device = device

        # Initialize hyperparameters
        self._init_hyperparameters(conf)

        # Extract environment information
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # ALG STEP 1
        # Initialize actor and critic networks
        self.actor = Policy(self.obs_dim, self.act_dim).to(device)
        self.critic = Value(self.obs_dim).to(device)

        # Initialize actor optimizer
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        # Initialize critic optimizer
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # Initialize critic criterion
        self.critic_criterion = nn.MSELoss().to(device)

        # Create our variable for the matrix.
        # Note that I chose 0.5 for stdev arbitrarily.
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5, device=device)

        # Create the covariance matrix
        self.cov_mat = torch.diag(self.cov_var).to(device)

        # This logger will help us with printing out summaries of each iteration
        self.logger = {
                'delta_t': time.time_ns(),
                't_so_far': 0,          # timesteps so far
                'i_so_far': 0,          # iterations so far
                'batch_lens': [],       # episodic lengths in batch
                'batch_rews': [],       # episodic returns in batch
                'actor_losses': [],     # losses of actor network in current iteration
        }

    def learn(self, total_timesteps):
        t_so_far = 0 # Timesteps simulated so far
        i_so_far = 0 # Iterations ran so far
        pbar = tqdm(total=total_timesteps)
        while t_so_far < total_timesteps: # ALG STEP 2
            # ALG STEP 3
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

            # Calculate how many timesteps we collected this batch
            delta_t = np.sum(batch_lens)
            t_so_far += delta_t
            pbar.update(delta_t)

            # Increment the number of iterations
            i_so_far += 1

            # Logging timesteps so far and iterations so far
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            self.ppo_update(batch_obs, batch_acts, batch_log_probs, batch_rtgs)

            # Print a summary of our training so far
            self._log_summary()

            # Save our model if it's time
            if i_so_far % self.save_freq == 0:
                torch.save(self.actor.state_dict(), './models_ppo/actor' + str(i_so_far))
                torch.save(self.critic.state_dict(), './models_ppo/critic' + str(i_so_far))

    def ppo_update(self, batch_obs, batch_acts, batch_log_probs, batch_rtgs):
        # Calculate V_{phi, k}
        V, _ = self.evaluate(batch_obs, batch_acts)

        # ALG STEP 5
        # Calculate advantage
        A_k = batch_rtgs - V.detach()

        # Normalize advantages
        A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

        # ALG STEP 6 & 7
        for _ in range(self.n_updates_per_iteration):
            # Calculate pi_theta(s_t | s_t)
            V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

            # Calculate ratios
            ratios = torch.exp(curr_log_probs - batch_log_probs)

            # Calculate surrogate losses
            surr1 = ratios * A_k
            surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k
            # Calculate actor loss
            actor_loss = -((torch.min(surr1, surr2) + self.Lambda * ratios * curr_log_probs).mean())

            # Calculate gradients and perform backward propagation for actor network
            self.actor_optim.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optim.step()

            # Calculate critic loss
            critic_loss = self.critic_criterion(V, batch_rtgs)

            # Calculate gradients and perform backward propagation for critic network
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            # Log actor loss
            self.logger['actor_losses'].append(actor_loss.detach().to(torch.device('cpu')))

    def evaluate(self, batch_obs, batch_acts):
        # Query critic network for a value V for each obs in batch_obs
        V = self.critic(batch_obs).squeeze()

        # Calculate the log probabilities of batch actions using most recent actor network.
        # This segment of code is similar to that in get_action()
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        # Return predicted values V and log probs log_probs
        return V, log_probs

    def rollout(self):
        # Batch data
        batch_obs = [] # batch observations
        batch_acts = [] # batch actions
        batch_log_probs = [] # log probs of each action
        batch_rews = [] # batch rewards
        batch_rtgs = [] # batch rewards-to-go
        batch_lens = [] # episodic lengths in batch

        self.actor = self.actor.to(torch.device('cpu'))
        self.cov_mat = torch.diag(self.cov_var).to(torch.device('cpu'))

        # Number of timestpes run so far this batch
        t = 0
        while t < self.timesteps_per_batch:
            # Rewards this episode
            ep_rews = []
            obs = self.env.reset()
            done = False
            for ep_t in range(self.max_timesteps_per_episode):
                # Increment timesteps ran this batch so far
                t += 1
                # Collect observation
                batch_obs.append(obs)
                action, log_prob = self.get_action(obs)
                obs, rew, done, _ = self.env.step(action)

                # Collect reward, action, and log prob
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break

            # Collect episodic length and rewards
            batch_lens.append(ep_t + 1) # plus 1 because timestep starts at 0
            batch_rews.append(ep_rews)

        self.actor = self.actor.to(torch.device(self.device))
        self.cov_mat = torch.diag(self.cov_var).to(self.device)

        # Reshape data as tensors in the shape specified before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float, device=self.device)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float, device=self.device)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float, device=self.device)
        # ALG STEP #4
        batch_rtgs = self.compute_rtgs(batch_rews).to(self.device)

        # Log the episodic returns and episodic lengths in this batch.
        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens

        # Return the batch data
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def get_action(self, obs):
        # Query the actor network for a mean action.
        # Same thing as calling self.actor.forward(obs)
        mean = self.actor(torch.tensor(obs, dtype=torch.float))
        # Create our Multivariate Normal Distribution
        dist = MultivariateNormal(mean, self.cov_mat)
        # Sample an action from the distribution and get its log prob
        action = dist.sample()
        log_prob = dist.log_prob(action)
        # Return the sampled action and the log prob of that action
        # Note that I'm calling detach() since the action and log_prob  
        # are tensors with computation graphs, so I want to get rid
        # of the graph and just convert the action to numpy array.
        # log prob as tensor is fine. Our computation graph will
        # start later down the line.
        return action.detach().numpy(), log_prob.detach()

    def compute_rtgs(self, batch_rews):
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []
        # Iterate through each episode backwards to in batch_rtgs
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0 # The discounted reward so far
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs

    def _init_hyperparameters(self, conf):
        # Default values for hyperparameters, will need to change later.
        if 'timesteps_per_batch' in conf: # timesteps per batch
            self.timesteps_per_batch = conf['timesteps_per_batch']
        else:
            self.timesteps_per_batch = 2048
        if 'max_timesteps_per_episode' in conf: # timesteps per episode
            self.max_timesteps_per_episode = conf['max_timesteps_per_episode']
        else:
            self.max_timesteps_per_episode = 200
        if 'gamma' in conf: # discount factor
            self.gamma = conf['gamma']
        else:
            self.gamma = 0.99
        if 'n_updates_per_iteration' in conf: # number of epochs per iteration
            self.n_updates_per_iteration = conf['n_updates_per_iteration']
        else:
            self.n_updates_per_iteration = 10
        if 'clip' in conf: # clip threshold as recommended by the paper
            self.clip = conf['clip']
        else:
            self.clip = 0.2
        if 'lr' in conf: # learning rate of optimizers
            self.lr = conf['lr']
        else:
            self.lr = 3e-4
        if 'save_freq' in conf:
            self.save_freq = conf['save_freq']
        else:
            self.save_freq = 10
        if 'lammbda' in conf:
            self.Lambda = conf['lambda']
        else:
            self.Lambda = 0.001

    def _log_summary(self):
            """
                    Print to stdout what we've logged so far in the most recent batch.
                    Parameters:
                            None
                    Return:
                            None
            """
            # Calculate logging values. I use a few python shortcuts to calculate each value
            # without explaining since it's not too important to PPO; feel free to look it over,
            # and if you have any questions you can email me (look at bottom of README)
            delta_t = self.logger['delta_t']
            self.logger['delta_t'] = time.time_ns()
            delta_t = (self.logger['delta_t'] - delta_t) / 1e9
            delta_t = str(round(delta_t, 2))

            t_so_far = self.logger['t_so_far']
            i_so_far = self.logger['i_so_far']
            avg_ep_lens = np.mean(self.logger['batch_lens'])
            avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
            avg_actor_loss = np.mean([losses.to(torch.device('cpu')).float().mean() for losses in self.logger['actor_losses']])

            # Round decimal places for more aesthetic logging messages
            avg_ep_lens = str(round(avg_ep_lens, 2))
            avg_ep_rews = str(round(avg_ep_rews, 2))
            avg_actor_loss = str(round(avg_actor_loss, 5))

            # Print logging statements
            print(flush=True)
            print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
            print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
            print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
            print(f"Average Loss: {avg_actor_loss}", flush=True)
            print(f"Timesteps So Far: {t_so_far}", flush=True)
            print(f"Iteration took: {delta_t} secs", flush=True)
            print(f"------------------------------------------------------", flush=True)
            print(flush=True)

            # Reset batch-specific logging data
            self.logger['batch_lens'] = []
            self.logger['batch_rews'] = []
            self.logger['actor_losses'] = []
