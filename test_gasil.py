from gasil import *
from eval_policy import *
import gym


env = gym.make('Hopper-v2')
model = GASIL(env)
model.actor.load_state_dict(torch.load('models_gasil/actor'))
eval_policy(model.actor, env, True)
