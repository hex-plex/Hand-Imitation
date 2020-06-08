import pybullet as p
import time
import os

pclient = p.connect(p.GUI)
p.setAdditionalSearchPath(os.path.abspath("Simulation"))
hand = p.loadURDF("hand.urdf")

for i in range(p.getNumJoints(hand)):
    print(p.getJointInfo(hand, i))

for i in range(1000):
    time.sleep(0.01)
    p.stepSimulation()

p.disconnect()
