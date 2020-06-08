import gym
from gym import error, spaces, utils
from gym.utils import seeding

import pybullet as p
import pybullet_data
import numpy as np
import cv2
import hand_controller as hc

class HandOfJustice(gym.Env):
    metadata = {'render.modes':['human']}

    def __init__(self,mod="Direct"):
        self.cap = cv2.VideoCapture(0) ## Set if a particular usb cam is used
        if mod == "GUI":
            p.connect(p.GUI)
            p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0,0,2])
            ## This is just to see the hand through opengl window so hence set this as  you see the hand as you want to see
        else:
            p.connect(p.DIRECT)

        
        self.action_space = spaces.Box(low=[0]*12 ,high=[1]*12) ## Have to edit based on yashs input
        self.observation_space = spaces.Box(0,2.55,shape=(56,56,3))## remember to rescale
        ## Remember to change this
        ## Initilize the hand
        self.threshold=150
        ## THis is to match up the no of pixels of our PHATTTT
        
        

    def step(self,action):
        armCam=np.zeros(56,56,3)## take in an input from the pybullet camera
        ## set action as the joint angles
        error = np.sum(np.abs(armCam-self.target)) ## Remember to convert BGR 2 RGB
        ## Try a thresholded or a mask rather than the whole thing
        if error<=self.threshold:
            done = True
        else:
            done = False
        return self.target, -error , done, {}
        ## End point is anypoint with a error of a Threshold
    def reset(self):
        p.resetSimulation()
        p.configureDebugVisualiser(p.COV_ENABLE_RENDERING,0)
        p.setGravity(0,0,-10)
        ## Initilize the hand same like the one done in __init__
        ## Or just call to reset to that point
        ## This can be skipped if a continues feel is to be got
        self.target = self.cap.read()[1]     
        ## Take in a feed
        ## observation is the target image
        return self.target
    
    def render(self,mode='human'):
        armCam=np.zeros(56,56,3)## take in an input from the pybullet camera
        ## a higher resolution can be used
        return armCam
        
    def close(self):
        p.disconnect()
    
