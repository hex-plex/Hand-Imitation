import numpy as np
import pybullet as p
import gym
import gym_handOfJustice
import cv2

env = gym.make('handOfJustice-v0',cap = cv2.VideoCapture(0),mod="GUI",epsilon=200)

for i in range(5000):
    observation,reward,done,_=env.step(np.random.rand(12))
    print(reward)
    robo = env.getImage()
    #robo = cv2.cvtColor(robo,cv2.COLOR_BGR2GRAY)
    #_,robo = cv2.threshold(robo,185,255,cv2.THRESH_BINARY_INV)
    print(p.isConnected(env.clientId))
    handthr = env.hand_thresh(env.target)
    cv2.imshow("robo",robo)
    cv2.imshow("hand",handthr)
    cv2.imshow("poiu",observation)
    cv2.waitKey(1)
    if i%200==0:
        env.reset()
