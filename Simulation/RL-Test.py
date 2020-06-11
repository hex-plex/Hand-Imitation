import gym
import cv2
from stable_baselines.sac import SAC
from stable_baselines.sac.policies import LnCnnPolicy
env = gym.make("handOfJustice-v0",cap=cv2.VideoCapture,mod="GUI",threshold=300)
                                            #arguements dont train in mod = "GUI"
                                            #all of these are defaulted
model = SAC(LnCnnPolicy, env , verbose=1,tensorboard_log="./logs")
model.learn(total_timesteps=50000,log_interval=10)
model.save("handicap_justice")

obs = env.reset()
while True:
    action,states = model.predict(obs)
    obs, rewards,dones,info = env.step(action)
    env.render()
