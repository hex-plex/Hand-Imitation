import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.layers import Input,Dense,Conv2D,MaxPooling2D,Flatter
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras.application.mobilenet_v2 as MobileNetv2

data={
    'clip_val':0.2,
    'critic_dis':0.5,
    'gamma':0.99,
    'lambda':0.65
    }
