import gym
from gym import error, spaces, utils
from gym.utils import seeding

import pybullet as p
import pybullet_data
import numpy as np
import cv2
import hand_controller as hc
from hand_mask import handmask

class HandOfJustice(gym.Env):
    metadata = {'render.modes':['human']}

    def __init__(self,cap=cv2.VideoCapture(0),mod="Direct",threshold=150):
        self.cap =  cap
        if mod == "GUI":
            p.connect(p.GUI)
            p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0,0,2])
            ## This is just to see the hand through opengl window so hence set this as  you see the hand as you want to see
        else:
            p.connect(p.DIRECT)


        self.action_space = spaces.Box(low=[0]*10+[-0.52,-1.04] ,high=[1.55]*10+[0.52,1.04])
        ## down and up (thumb, index, middle, ring, little) , wrist, elbow
        self.observation_space = spaces.Box(0,2.55,shape=(56,56,3))## remember to rescale
        ## Remember to change this
        p.setAdditionalSearchPath(os.path.abspath("Simulation"))
        self.handid = p.loadURDF("hand.urdf")

        self.threshold=threshold ## Find a good one and set as default
        self.seed(int(time.time()))
        ## THis is to match up the no of pixels of our PHATTTT
        p.createConstraint(
            handid,
            -1,
            -1,
            -1,
            p.JOINT_FIXED,
            (0, 0, 1),
            (0, 0, 0),
            (0, 0, 0),
        )
        finger_joint_indices = (
            (2, 3),  # thumb
            (4, 5),  # index
            (6, 7),  # middle
            (8, 9),  # ring
            (10, 11),  # little
        )
        self.hand = hc.robo_hand(handid,finger_joint_indices)


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def getImage(flag=True):
        ################################################
        ## Atul edit your pybullet preprocessing here ##
        ################################################
        position = (0, -3, 2)
        targetPosition = (0, 0, 2)
        viewMatrix = p.computeViewMatrix(
            position, targetPosition, cameraUpVector=[0, 0, 1])
        projectionMatrix = p.computeProjectionMatrixFOV(60, 1, 0.02, 5)
        img = p.getCameraImage(56, 56, viewMatrix, projectionMatrix,
                           renderer=p.ER_BULLET_HARDWARE_OPENGL)
        img = np.reshape(img[2], (56, 56, 4))
        ## make this of only 3 channels no need of the last one
        if flag:
            img=cv2.imread('handimg.png',0)
            new_img=cv2.resize(img,(512,512))
            _,thresh = cv2.threshold(new_img,127,255,cv2.THRESH_BINARY_INV)
            cv2.imshow('thresholded_bot',thresh)
            #pass
        else:
            ## nopreprocess
            pass

        return img.astype('uint8')

    def step(self,action):
        armCam=self.getImage()
        ## Preprocess the image
        self.hand.array_input(list(list([action[2*i],action[(2*i)+1]] for i in range(5))+[action[10],action[11]]))
        error = np.sum(np.abs(armCam-self.target))
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
        cv2.waitKey(1)
        self.target = handmask(self.target)
        return self.target

    def render(self,mode='human'):
        armCam=self.getImage(flag=True)
        ## a higher resolution can be used
        return armCam

    def close(self):
        p.disconnect()
