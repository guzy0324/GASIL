import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from queue import PriorityQueue

from good_trajectory import GoodTrajectory
from mlp_discriminator import Discriminator
from ppo import PPO


class GASIL(PPO):
    def __init__(self, env, conf=dict(), device=torch.device('cpu')):
        super(GASIL, self).__init__(env, conf, device)

        # ALG STEP 1
        # Initialize discriminator network
        self.D = Discriminator(self.obs_dim + self.act_dim).to(device)

        # Initialize discriminator optimizer
        self.D_optim = Adam(self.D.parameters(), lr=self.lr)

        # Initialize discriminator criterion
        self.D_criterion = nn.BCELoss()

        # Initialize good trajectory buffer B
        self.B = PriorityQueue()

    def learn(self, total_timesteps):
        t_so_far = 0 # Timesteps simulated so far
        i_so_far = 0 # Iterations ran so far
        pbar = tqdm(total=total_timesteps)
        while t_so_far < total_timesteps: # ALG STEP 2
            # ALG STEP 3
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_E_obs, batch_E_acts = self.rollout()

            # Calculate how many timesteps we collected this batch
            delta_t = np.sum(batch_lens)
            t_so_far += delta_t
            pbar.update(delta_t)

            # Increment the number of iterations
            i_so_far += 1

            # Logging timesteps so far and iterations so far
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            # Update the discriminator \phi via gradient ascent with:
            first_g_o = None
            for _ in range(self.n_updates_of_D_per_iteration):
                g_o = self.D(torch.cat([batch_obs, batch_acts], 1))
                if first_g_o is None:
                    first_g_o = g_o.squeeze(1).detach()
                e_o = self.D(torch.cat([batch_E_obs, batch_E_acts], 1))
                self.D_optim.zero_grad()
                discrim_loss = - self.D_criterion(g_o, torch.ones((batch_obs.shape[0], 1), device=self.device)) - \
                    self.D_criterion(e_o, torch.zeros((batch_E_obs.shape[0], 1), device=self.device))
                discrim_loss.backward()
                self.D_optim.step()

            # Modified reward function
            batch_rtgs = batch_rtgs - self.alpha * torch.log(first_g_o)

            self.ppo_update(batch_obs, batch_acts, batch_log_probs, batch_rtgs)

            # Print a summary of our training so far
            self._log_summary()

            # Save our model if it's time
            if i_so_far % self.save_freq == 0:
                torch.save(self.actor.state_dict(), './models_gasil/actor' + str(i_so_far))
                torch.save(self.critic.state_dict(), './models_gasil/critic' + str(i_so_far))
                torch.save(self.D.state_dict(), './models_gasil/D' + str(i_so_far))

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

            # Update good trajectory buffer B using \Tau_\pi
            self.B.put(GoodTrajectory(batch_obs[t - ep_t - 1:t], batch_acts[t - ep_t - 1:t], sum(ep_rews)))
            if self.B.qsize() > self.K:
                self.B.get()

        self.actor = self.actor.to(self.device)
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

        # Sample good trajectories \Tau_E ~ B
        B = [self.B.get() for _ in range(self.B.qsize())]
        batch_E = np.random.choice(B, size=len(batch_lens), replace=True)
        for E in B:
            self.B.put(E)

        # Reshape data as tensors in the shape specified before returning
        batch_E_obs = list()
        batch_E_acts = list()
        for E in batch_E:
            batch_E_obs.extend(E.obs)
            batch_E_acts.extend(E.acts)
        batch_E_obs = torch.tensor(batch_E_obs, dtype=torch.float, device=self.device)
        batch_E_acts = torch.tensor(batch_E_acts, dtype=torch.float, device=self.device)

        # Return the batch data
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_E_obs, batch_E_acts
    
    def _init_hyperparameters(self, conf):
        # Default values for hyperparameters, will need to change later.
        super(GASIL, self)._init_hyperparameters(conf)

        if 'n_updates_of_D_per_iteration' in conf: # number of updates of discriminator per iteration
            self.n_updates_of_D_per_iteration = conf['n_updates_of_D_per_iteration']
        else:
            self.n_updates_of_D_per_iteration = 20
        if 'K' in conf: # size of good trajectory buffer B
            self.K = conf['K']
        else:
            self.K = 1000
        if 'alpha' in conf: # weight of log(D)
            self.alpha = conf['alpha']
        else:
            self.alpha = 1
