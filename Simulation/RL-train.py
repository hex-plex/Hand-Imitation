import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.layers import Input,Dense,Conv2D,MaxPooling2D,Flatter
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras.application.mobilenet_v2 as MobileNetv2

hyperparam={
    'clip_val':0.2,
    'critic_dis':0.5,
    'entrophy_beta':0.001,
    'gamma':0.99,
    'lambda':0.65,
    'actor_lr': 1e-4,
    'critic_lr': 1e-3
    }

def advantages(valueS,masks,rewardSA):
    returns=[]
    gae=0
    for i in range(len(rewardSA)-1,-1,-1):  
        delta = rewardSA[i] + hyperparam['gamma']*valueS[i+1]*masks[i] -valueS[i]
        gae = delta + hyperparam['gamma']*hyperparam['lambda']*masks[i]*gae
        returns.append(gae+valueS[i])
    returns=returns[::-1]  ## Check If THis WOrks

    adv = np.array(returns) - valueS[:-1]
    return returns, ((adv -np.mean(adv))/(np.std(adv)+1e-10))

def ppo_loss_actor(old_policy_probs,advantages,rewards,valueS):
    def loss(y_true,y_pred):
        y_true = tf.Print(y_true,[y_true],'y_true: ')
        y_pred = tf.Print(y_pred,[y_pred],'y_pred: ')
        new_policy_probs = y_pred # Check Here also
        new_policy_probs = tf.Print(new_policy_probs, [new_policy_probs],'new_policy_probs: ')
        ratio = K.exp(K.log(new_policy_probs)- K.log(old_policy_probs) + 1e-10)
        ratio = tf.Print(ratio,[ratio],'ratio: ')
        p1 = ratio*advantages
        p2 = K.clip(ratio,min_value= 1-clipping_val,max_value=1+clipping_val)*advantages
        actor_loss = -K.mean(K.minimum(p1,p2))
        actor_loss = tf.Print(actor_loss , [actor_loss] ,'actor_loss: ')
        critic_loss = K.mean(K.square(rewards - valueS))
        critic_loss = tf.Print(critic_loss , [critic_loss],'critic_loss: ')
        term_al = hyperparam['critic_dis']*critic_loss
        term_al = tf.Print(term_al, [term_al], 'term_al: ')
        term_b2 = K.log(new_policy_probs + 1e-10)
        term_b2 = tf.Print(term_b2, [term_b2], 'term_b2: ')
        term_b = hyperparam['entrophy_beta']*K.mean(-(new_policy_probs*term_b2))
        term_b = tf.Print(term_b, [term_b], 'term_b')
        total_loss = term_a +actor_loss - term_b
        total_loss = tf.Print(total_loss, [total_loss],'total_loss: ')
        return total_loss
    return loss

def ppo_loss_np(old_policy_probs,advantages,rewards,valueS):
    def loss(y_true,y_pred):
        new_policy_probs = y_pred # Check Here also
        ratio = K.exp(K.log(new_policy_probs)- K.log(old_policy_probs) + 1e-10)
        p1 = ratio*advantages
        p2 = K.clip(ratio,min_value= 1-clipping_val,max_value=1+clipping_val)*advantages
        actor_loss = -K.mean(K.minimum(p1,p2))
        critic_loss = K.mean(K.square(rewards - valueS))
        term_al = hyperparam['critic_dis']*critic_loss
        term_b2 = K.log(new_policy_probs + 1e-10)
        term_b = hyperparam['entrophy_beta']*K.mean(-(new_policy_probs*term_b2))
        total_loss = term_a +actor_loss - term_b
        return total_loss
    return loss

def model_actor_image(input_dims, output_dims):
    state_input = Input(shape=input_dim)
    old_policy_probs = Input(shape=(1,output_dims,))
    advantages = Input(shape=(1,1,))
    rewards = Input(shape=(1,1,))
    valueS = Input(shape=(1,1,))
    feature_image = MobileNetV2(include_top=False, weights='imagenet')
    for layer in feature_image.layers:
        layer.trainable = False

    x = Flatten(name = 'flatten')(feature_image(state_input))
    x = Dense(1024,activation='relu' , name='fc1')(x)
    out_actions = Dense(n_actions,activation='softmax',name='predictions')(x)

    model = Model(inputs=[state_input, old_policy_probs, advantages,rewards, valueS],
                  outputs=[out_actions])
    model.compile(optimizer=Adam(lr=hyperparam['actor_lr']),loss=[ppo_loss_np(
                                                                        old_policy_probs=old_policy_probs,
                                                                        advantages=advantages,
                                                                        rewards=rewards,
                                                                        valueS=valueS)])
    model.summary()
    return model




tensor_board = TensorBoard(log_dir='./logs')
