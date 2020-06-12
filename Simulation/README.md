# Simulation
Here are all the files used in the simulation and learning process of the agent.

# HandOFJustice
This is a robotic arm which takes in a an input of 12 joint angles to Replicate a given pose.
The environment has a few parameter to customize:
- feed parameter where in an instance of cv2.VideoCapture is passed
- mod parameter This is simply to run a env with pybullet in GUI or in Direct (which is for DIRECT of pybullet to be used)
- epsilon parameter is for measuring how similar two images are, This helps in compensating  the excess length, width or thickness of one's hand with respect to the robotic arm.
- preprocess parameter takes in a function that helps to threshold one's hand, this is one of the most important one as we are using traditional methods to process your hand for a reward function,And it might not suite all lighting conditions or hand types (Even with our trys to make it as robust as possible ).
for passing a function as preprocess which should threshold your hand and return a mask of your hand
- resolution parameter is for defining the resolution of the hand image itself or the resolution to which the hand image is resized to be used and outputed (remember to keep the aspect ratio of both same).
 
for more info, source and doc [gym-handOFJustice](https://github.com/hex-plex/gym-handOfJustice)

>pip install gym-handOFJustice
## Example
```python
import gym
import gym_handOfJustice
import cv2

env = gym.make('handOFJustice-v0',cap=cv2.VideoCapture(1), mod="GUI",epsilon=150,preprocess=None,resolution=(56,56,3))
```

## Reward function
For the environment reward is the negative of no of pixels that doesnt match up in the two thresholded images
hence we might need a epsilon to define the acceptable amount of error for a reward to interpret a state as a terminal state of an episode.

# Reinforcement learning Algorithm
Our primary intension was to train a CNN which could predict the set of 12 Joint angles based on the given feed of a real hand,
The means to train it was through *Actor - Critic* Algorithm where in the error to backpropagate through the CNN is the return we get for taking an action with respect to a baseline in this case is the previous expectation of return for that given state(Value Function of the State)
For Training we run RL-train.py
>python RL-train.py

# Updates Required
- [ ] Setup a auto cropping mechanism for a image based on cascading algorithms
- [ ] Use a PID controller to go from a current pose to another as its what would be used in real world
- [ ]  u