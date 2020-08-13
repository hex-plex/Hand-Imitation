import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Input,Dense,Conv2D,MaxPooling2D,Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import numpy as np
import os
import gym
import pybullet
import cv2
import gym_handOfJustice

hyperparam={
    'clip_val':0.2,
    'critic_dis':0.5,
    'entrophy_beta':0.001,
    'gamma':0.99,
    'lambda':0.65,
    'actor_lr': 1e-4,
    'critic_lr': 1e-3
    }

class CustomCallBack(tf.keras.callbacks.Callback):

    def __init__(self,log_dir=None,tag=''):
        if log_dir is None:
            raise Exception("No logging directory received")
        self.writer = tf.summary.FileWriter(log_dir)
        self.step_number=0
        self.tag=tag
    def on_epoch_end(self,epoch,logs=None):
        item_to_write={
            'loss':logs['loss']
            }
        for name, value in item_to_write.items():
            summary = tf.summary.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = self.tag+name
            self.writer.add_summary(summary,self.step_number)
            self.writer.flush()

    def step_one(self):
        self.step_number +=1
    def __call__(self,tag):
        self.tag=tag
        return self
    
def model_critic_image(input_dims):
    state_input = Input(shape=input_dims,name="state_input")
    feature_image = MobileNetV2(include_top=False, weights='imagenet')
    for layer in feature_image.layers:
        layer.trainable=False
    x = Flatten(name='flatten_features')(feature_image(state_input))
    x = Dense(128,activation='relu',name='forwardc1')(x)
    x = Dense(32,activation='relu',name='forwardc2')(x)
    Value_function = Dense(1,name="value")(x)

    model = Model(inputs=[state_input],outputs=[Value_function])
    model.compile(optimizer=Adam(lr=hyperparam['critic_lr']),loss='mse')
    model.summary()
    return model


#tensor_board = TensorBoard()
custom_callback=CustomCallBack(log_dir=os.getcwd()+'\\logs\\Training\\')
strea = cv2.VideoCapture(os.getcwd()+"\\dataset\\%06d.png")
if not strea.isOpened():
    raise Exception("Problem exporting the video stream")
env = gym.make('handOfJustice-v0',cap=strea,epsilon=200)
state_dims = env.observation_space.shape
action_dims = env.action_space.shape

critic_model = model_critic_image(input_dims=state_dims)
#actor_json = actor_model.to_json()
#critic_json = critic_model.to_json()
#with open("actor_model.json","w") as json_f:
    #json_f.write(actor_json)
#with open("critic_model.json","w") as json_f:
    #json_f.write(critic_json) 
## Saving a json file is not possible as in tensorflow < 2 there is problem in having layers in same order
best_reward= -10000000000
num_episodes= 50000
episode_history=[]
batch_size = 25
for episode in range(num_episodes):
    state = env.reset()
    state = state.reshape((1,)+state.shape  )
    reward_total = 0
    step = 0
    done=False 
    while not done:
        action = np.squeeze(actor_model.predict([state,np.array([0])],steps=1))
        #print(action)
        for i in range(len(action)):
            action[i] = max(min(action[i],env.action_space.high[i]),env.action_space.low[i])
        next_state,reward,done,_=env.step(np.squeeze(action))
        next_state = next_state.reshape((1,) + next_state.shape )
        step+=1
        reward_total +=reward
        V_of_next_state = critic_model.predict(next_state)
        V_this_state = critic_model.predict(state)
        target = reward + hyperparam['gamma']*np.squeeze(V_of_next_state)
        td_error = target - np.squeeze(V_this_state)
        td_error = np.array([td_error])
        critic_model.fit([state],[target],callbacks=[custom_callback('critic')])
        state=next_state
        custom_callback.step_one()
    episode_history.append(reward_total)
    print("Episode: {}, Number of Steps : {}, Cumulative reward: {:0.2f}".format(
            episode, step, reward_total))
    if (episode+1)%5000:
        critic_model.save_weights("checkpoints/critic_model-"+str(episode)+".h5")
