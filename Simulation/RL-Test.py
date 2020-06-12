import gym
import gym_handOfJustice
import cv2
from stable_baselines.sac import SAC
from stable_baselines.sac.policies import LnCnnPolicy
env = gym.make("handOfJustice-v0",cap=cv2.VideoCapture(<--directory-->),epsilon=150)

model = SAC(LnCnnPolicy, env , verbose=1,tensorboard_log="./logs")
model.learn(total_timesteps=50000,log_interval=10)
model.save("handicap_justice")

import time;time.sleep(3)

## No is the first image gonna be taken
obs = env.reset()
while True:
    action,states = model.predict(obs)
    obs, rewards,dones,info = env.step(action)
    ## Dont keep it on while training but i dont think that will be the case
    cv2.imshow("This is what the robotic arm is doing",env.render())
    cv2.waitKey(1)
    time.sleep(0.05)
