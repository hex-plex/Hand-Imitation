import gym
from gym import error, spaces, utils
from gym.utils import seeding

import pybullet as p
import pybullet_data
import numpy as np
import cv2
import os
import inspect
import time

import pkg_resources

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)


class finger():
    lower = None
    upper = None

    def __init__(self, lower_joint, upper_joint,handid,clientId):
        self.lower = lower_joint
        self.upper = upper_joint
        self.handid = handid
        self.clientId=clientId

    def rotate(self, lower_angle, upper_angle):
        p.setJointMotorControlArray(bodyIndex=self.handid,
                                    jointIndices=(self.lower, self.upper),
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=(lower_angle, upper_angle),
                                    forces=[500, 500],
                                    physicsClientId=self.clientId)
        p.stepSimulation(physicsClientId=self.clientId)


class robo_hand():
    fingers = list()
    elbow_index = 0
    wrist_index = 1

    def __init__(self,handid,finger_joint_indices,clientId):
        
        self.handid=handid
        self.finger_joint_indices = finger_joint_indices
        self.clientId=clientId
        self.wave_arm(0)
        for indices in self.finger_joint_indices:
            self.fingers.append(finger(*indices,self.handid,self.clientId))

    def fold_finger(self, index, lower_angle, upper_angle):
        self.fingers[index].rotate(lower_angle, upper_angle)
    
    def wave_arm(self, angle):
        # p.resetJointState(
        #     bodyUniqueId=self.handid,
        #     jointIndex=self.elbow_index,
        #     targetValue=angle,
        # )
        p.setJointMotorControl2(
            bodyIndex = self.handid,
            jointIndex = self.elbow_index,
            controlMode = p.POSITION_CONTROL,
            targetPosition = angle,
            force = 0.5,
            maxVelocity = 0.4,
            physicsClientId=self.clientId
        )
        p.stepSimulation(physicsClientId=self.clientId)

    def move_wrist(self, angle):
        p.setJointMotorControl2(
            bodyIndex = self.handid,
            jointIndex = self.wrist_index,
            controlMode = p.POSITION_CONTROL,
            targetPosition = angle,
            force = 0.5,
            maxVelocity = 0.4,
            physicsClientId=self.clientId
        )
        p.stepSimulation(physicsClientId=self.clientId)
    def array_input(self,arr):
        for i in range(5):
            self.fingers[i].rotate(*arr[i])
        self.move_wrist(arr[5])
        self.wave_arm(arr[6])
        #p.stepSimulation(physicsClientId=self.clientId)




class HandOfJusticeEnv(gym.Env):
    metadata = {'render.modes':['human']}

    def __init__(self,cap=cv2.VideoCapture(0),mod="Direct",epsilon=150,preprocess=None,resolution=(56,56,3)):
        self.cap =  cap
        if mod == "GUI":
            self.clientId = p.connect(p.GUI)
            ## This is just to see the hand through opengl window so hence set this as  you see the hand as you want to see
        else:
            self.clientId = p.connect(p.DIRECT)

        
        #p.setRealTimeSimulation(1,physicsClientId=self.clientId)
        #p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0,0,2],physicsClientId=self.clientId)
            
        self.action_space = spaces.Box(low=np.array([0]*10+[-0.52,-1.04]) ,high=np.array([1.55]*10+[0.52,1.04]))
        ## down and up (thumb, index, middle, ring, little) , wrist, elbow

        if len(resolution)!=3:
            raise Exception("Only a ndim n=3 image can be given as a input")

        self.res=resolution
        self.observation_space = spaces.Box(0,2.55,shape=(self.res[0],self.res[1]*2,self.res[2]))
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane = p.loadURDF( "plane.urdf" , physicsClientId=self.clientId)
        p.setAdditionalSearchPath(os.path.abspath("Simulation"))
        self.handid = p.loadURDF(currentdir+"/hand.urdf",physicsClientId=self.clientId)
        for i in range(p.getNumJoints(self.handid,physicsClientId=self.clientId)):
            print(p.getJointInfo(bodyUniqueId=self.handid,jointIndex=i,physicsClientId=self.clientId))
        if preprocess is None:
            self.hand_thresh=self.handmask
        else:
            self.hand_thresh=preprocess
        self.epsilon=epsilon ## Find a good one and set as default
        self.seed(int(time.time()))
        ## THis is to match up the no of pixels of our PHATTTT
        p.createConstraint(
            self.handid,
            -1,
            -1,
            -1,
            p.JOINT_FIXED,
            (0, 0, 1),
            (0, 0, 0),
            (0, 0, 0),
            physicsClientId=self.clientId
        )
        finger_joint_indices = (
            (2, 3),  # thumb
            (4, 5),  # index
            (6, 7),  # middle
            (8, 9),  # ring
            (10, 11),  # little
        )
        self.hand = robo_hand(self.handid,finger_joint_indices,clientId=self.clientId)
        self.resetState = p.saveState()
        self.reset()


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def getImage(self,flag=False):
        position = (0, -3, 2)
        targetPosition = (0, 0, 2)
        viewMatrix = p.computeViewMatrix(
            position, targetPosition, cameraUpVector=[0, 0, 1],
            physicsClientId=self.clientId)
        projectionMatrix = p.computeProjectionMatrixFOV(60, 1, 0.1, 3.5,physicsClientId=self.clientId)
        img = p.getCameraImage(self.res[0], self.res[1], viewMatrix, projectionMatrix,
                           renderer=p.ER_BULLET_HARDWARE_OPENGL,
                               physicsClientId=self.clientId)
        img = np.reshape(img[2], (self.res[0],self.res[1], 4))
        
        if flag:
            img = img[:,:,:3]
        else:
            img = img[:,:,:3]
            img= cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            _,img = cv2.threshold(img,180,255,cv2.THRESH_BINARY_INV)

        return img.astype('uint8')

    def handmask(self,frame):
        frame=cv2.flip(frame,1)
        kernel = np.ones((3,3),np.uint8)
        #cv2.rectangle(frame,(100,100),(300,400),(0,255,0),0)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lower_skin = np.array([40,120,120], dtype=np.uint8)
        upper_skin = np.array([255,170,170], dtype=np.uint8)
        mask = cv2.inRange(lab, lower_skin, upper_skin)
        #cv2.imshow('mask',mask)   ## This would also need a waitkey to work
        #cv2.imshow('frame',frame)  ## THis would as crash ones computer as ram is not more that 16 gb in a normal computer
        #cv2.imshow("cropped",cr_frame)
        return mask
    
    def step(self,action):
        #print(armCam.shape)
        #print(tuple(list((action[2*i],action[(2*i)+1]) for i in range(5))+[action[10],action[11]])) 
        self.hand.array_input(tuple(list((action[2*i],action[(2*i)+1]) for i in range(5))+[action[10],action[11]]))
        p.stepSimulation(physicsClientId=self.clientId)
        armCam=self.getImage()
        robo = armCam>100
        handthr = self.hand_thresh(self.target) > 100
        u = robo^handthr
        error = np.sum(u)
        if error<=self.epsilon:
            done = True
        else:
            done = False

        if self.noofrun>1000:
            done=True
        self.noofrun+=1
        armCam=self.getImage(flag=True)
        return np.append(self.target,armCam,axis=1), -1*error , done, {}

    def reset(self):
        p.restoreState(self.resetState)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0,physicsClientId=self.clientId)
        p.setGravity(0,0,-10)
        self.noofrun=0
        ## Initilize the hand same like the one done in __init__
        ## Or just call to reset to that point
        ## This can be skipped if a continues feel is to be got


        self.target = self.cap.read()[1]
        try:
            self.target = cv2.resize(self.target,(self.res[0],self.res[1]))
        except:
            print("found ",self.target.size)
            raise Exception("the aspect tatio of the resolution and the given image doesnt match up")

        return np.append(self.target,self.getImage(flag=True),axis=1)

    def render(self,mode='human'):
        armCam=self.getImage(flag=True)
        return armCam

    def close(self):
        p.disconnect()
