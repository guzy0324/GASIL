from ppo import *
import gym


env = gym.make('Pendulum-v0')
model = PPO(env)
model.learn(10000)
torch.save(model.actor.state_dict(), 'models/a')
torch.save(model.critic.state_dict(), 'models/c')
