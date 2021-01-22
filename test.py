from ppo import *
from eval_policy import *
import gym


env = gym.make('Pendulum-v0')
model = PPO(env)
model.actor.load_state_dict(torch.load('a'))
model.critic.load_state_dict(torch.load('c'))
eval_policy(model.actor, env, True)
