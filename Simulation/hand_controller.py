import pybullet as p
import time
import os
from matplotlib import pyplot as plt
import numpy as np

pclient = p.connect(p.GUI)
p.setAdditionalSearchPath(os.path.abspath("Simulation"))
handid = p.loadURDF("hand.urdf")
p.setRealTimeSimulation(1)

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

    def rotate(self, angle):
        p.setJointMotorControlArray(bodyIndex=handid,
                                    jointIndices=(self.lower, self.upper),
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=(angle, angle),
                                    forces=[500, 500])


class robo_hand():
    fingers = list()

    def __init__(self):
        for indices in finger_joint_indices:
            self.fingers.append(finger(*indices))

    def fold_finger(self, index, angle):
        self.fingers[index].rotate(angle)


if __name__ == '__main__':
    hand = robo_hand()
    while True:
        # thumb
        hand.fold_finger(0, 1.5)
        time.sleep(1)
        # index
        hand.fold_finger(1, 1.5)
        time.sleep(1)
        # middle
        hand.fold_finger(2, 1.5)
        time.sleep(1)
        # ring
        hand.fold_finger(3, 1.5)
        time.sleep(1)
        # little
        hand.fold_finger(4, 1.5)
        time.sleep(1)

        # thumb
        hand.fold_finger(0, 0)
        time.sleep(1)
        # index
        hand.fold_finger(1, 0)
        time.sleep(1)
        # middle
        hand.fold_finger(2, 0)
        time.sleep(1)
        # ring
        hand.fold_finger(3, 0)
        time.sleep(1)
        # little
        hand.fold_finger(4, 0)
        time.sleep(1)
