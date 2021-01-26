from ppo import *
from eval_policy import *
import gym


env = gym.make('Hopper-v2')
model = PPO(env)
model.actor.load_state_dict(torch.load('models_ppo/actor'))
eval_policy(model.actor, env, True)
