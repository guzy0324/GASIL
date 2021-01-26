from ppo import *
import gym


env = gym.make('Hopper-v2')
conf = dict()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = PPO(env, conf, device)
model.learn(200000000)
