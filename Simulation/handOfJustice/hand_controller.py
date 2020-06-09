import pybullet as p
import time
import os
from matplotlib import pyplot as plt
import numpy as np



def getImage():
    """
    this function returns hand image in RGBA format
    """

    position = (0, -3, 2)
    targetPosition = (0, 0, 2)
    viewMatrix = p.computeViewMatrix(
        position, targetPosition, cameraUpVector=[0, 0, 1])
    projectionMatrix = p.computeProjectionMatrixFOV(60, 1, 0.02, 5)
    img = p.getCameraImage(512, 512, viewMatrix, projectionMatrix,
                           renderer=p.ER_BULLET_HARDWARE_OPENGL)
    img = np.reshape(img[2], (512, 512, 4))
    return img.astype('uint8')


class finger():
    lower = None
    upper = None

    def __init__(self, lower_joint, upper_joint):
        self.lower = lower_joint
        self.upper = upper_joint

    def rotate(self, lower_angle, upper_angle):
        p.setJointMotorControlArray(bodyIndex=handid,
                                    jointIndices=(self.lower, self.upper),
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=(lower_angle, upper_angle),
                                    forces=[500, 500])


class robo_hand():
    fingers = list()
    elbow_index = 0
    wrist_index = 1

    def __init__(self):
        self.wave_arm(0)
        for indices in finger_joint_indices:
            self.fingers.append(finger(*indices))

    def fold_finger(self, index, lower_angle, upper_angle):
        self.fingers[index].rotate(lower_angle, upper_angle)
    
    def wave_arm(self, angle):
        # p.resetJointState(
        #     bodyUniqueId=handid,
        #     jointIndex=self.elbow_index,
        #     targetValue=angle,
        # )
        p.setJointMotorControl2(
            bodyIndex = handid,
            jointIndex = self.elbow_index,
            controlMode = p.POSITION_CONTROL,
            targetPosition = angle,
            force = 0.5,
            maxVelocity = 0.4
        )

    def move_wrist(self, angle):
        p.setJointMotorControl2(
            bodyIndex = handid,
            jointIndex = self.wrist_index,
            controlMode = p.POSITION_CONTROL,
            targetPosition = angle,
            force = 0.5,
            maxVelocity = 0.4
        )
    
    def array_input(arr):
        assert len(arr)==12
        for i in range(5):
            self.fingers[i].rotate(*arr[i])
        self.move_wrist(*arr[5])
        self.wave_arm(*arr[6])


if __name__ == '__main__':
    pclient = p.connect(p.GUI)
    p.setAdditionalSearchPath(os.path.abspath("Simulation"))
    handid = p.loadURDF("hand.urdf")
    p.setRealTimeSimulation(1)
    for i in range(p.getNumJoints(handid)):
        print(p.getJointInfo(bodyUniqueId=handid,jointIndex=i))
          
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
    hand = robo_hand()
    while True:
        # wrist
        hand.move_wrist(0.4)
        time.sleep(1)
        # thumb
        hand.fold_finger(0, 1.5, 1.5)
        time.sleep(1)
        # index
        hand.fold_finger(1, 1.5, 1.5)
        time.sleep(1)
        # middle
        hand.fold_finger(2, 1.5, 1.5)
        time.sleep(1)
        # ring
        hand.fold_finger(3, 1.5, 1.5)
        time.sleep(1)
        # little
        hand.fold_finger(4, 1.5, 1.5)
        time.sleep(1)
        # elbow
        hand.wave_arm(0.5)
        time.sleep(1)
        # wrist
        hand.move_wrist(0)
        time.sleep(1)
        # thumb
        hand.fold_finger(0, 0, 0)
        time.sleep(1)
        # index
        hand.fold_finger(1, 0, 0)
        time.sleep(1)
        # middle
        hand.fold_finger(2, 0, 0)
        time.sleep(1)
        # ring
        hand.fold_finger(3, 0, 0)
        time.sleep(1)
        # little
        hand.fold_finger(4, 0, 0)
        time.sleep(1)
        # elbow
        hand.wave_arm(0.0)
        time.sleep(1)
