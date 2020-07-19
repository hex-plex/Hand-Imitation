import gym
import gym_handOfJustice
import cv2
import tensorflow as tf
from stable_baselines.sac import SAC
from stable_baselines.sac.policies import LnCnnPolicy
import os
strea = cv2.VideoCapture(os.getcwd()+"\\dataset\\%06d.png")
if not strea.isOpened():
    raise Exception("Problem exporting the video stream")
env = gym.make("handOfJustice-v0",cap=strea,epsilon=300)
#tf.test.is_gpu_available()
model = SAC(LnCnnPolicy, env , verbose=1,tensorboard_log=os.getcwd()+"\\logs\\",full_tensorboard_log=True)
model.load("handicap_justice")
model.learn(total_timesteps=100000,log_interval=10)
model.save("handicap_justice")
#model.load("handicap_justice")

import time;time.sleep(3)
print("\n"+("="*20)+"\nTraining complete\n"+("="*20)+"\n\n")
## No is the first image gonna be taken
obs = env.reset()
done = False
i=45000
try:
 while True:
    if done:
        i+=1
        if i>=49975:
            break
        obs = env.reset()
    action,states = model.predict(obs)
    obs, rewards,done,info = env.step(action)
    ## Dont keep it on while training but i dont think that will be the case
    cv2.imshow("This is what the robotic arm is doing",obs)
    cv2.waitKey(1)
    time.sleep(0.05)
except:
 cv2.destroyAllWindows()
