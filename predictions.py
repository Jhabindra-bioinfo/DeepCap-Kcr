#!/usr/bin/env python
# coding: utf-8

# ### This python code is for prediction, we do not label the data, we just want to know the the given sequence is either Kcr(positive) or Non-Kcr (Negative), If the model predict >0.5, which means the sequence contains Kcr modifications 

# ### Importing the python libraries, we used python version 3.7.4 and keras version 2.2.4 using TensorFlow backend

# In[1]:


import keras
from keras import backend as K
import tensorflow as tf
from keras import initializers,layers,regularizers
from keras.layers import Dropout
from keras import callbacks
from keras.models import *
from keras.engine.topology import Layer
from keras.callbacks import EarlyStopping
from keras.layers import Dropout,Activation
import numpy as np
from keras import optimizers
from Bio import SeqIO
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from scipy import interp
from keras.regularizers import l2
from keras.layers import Dense,Input,LSTM,Bidirectional,Activation,Conv1D,GRU,BatchNormalization,Concatenate


# In[3]:


import sys
sys.version


# In[4]:


import keras
keras.__version__ # keras version


# ### One-hot encoding or binary encoding for each amino acids

# In[6]:


def onehot(seq):
    bases = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
    X = np.zeros((len(seq),len(seq[0]),len(bases)))
    for i,m in enumerate(seq):
        for l,s in enumerate(m):
    #         print(s)
            if s in bases:
                X[i,l,bases.index(s)] = 1
    return X


# ### Function for reading a fasta file

# In[9]:


def read_fasta(file_path):
    '''File_path: Path to the fasta file
       Returns: List of sequence
    '''
    one=list(SeqIO.parse(file_path,'fasta'))
    return one


# ### capsul network codes start from here

# Our Capsnets code is modified version and origin is based on the keras implementation of CapsuleNet by Xifeng by Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Keras

# In[10]:


class Length(layers.Layer):
    """
    Compute the length of vectors. This is used to compute a Tensor that has the same shape with y_true in margin_loss
    inputs: shape=[dim_1, ..., dim_{n-1}, dim_n]
    output: shape=[dim_1, ..., dim_{n-1}]
    """

    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]


# ### Squash activation for   Equation 3,  described  in the main manuscript

# In[ ]:


def squash(vectors, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors


# In[11]:


class CapsuleLayer(layers.Layer):
    """
    The capsule layer. It is similar to Dense layer. Dense layer has `in_num` inputs, each is a scalar, the output of the
    neuron from the former layer, and it has `out_num` output neurons. CapsuleLayer just expand the output of the neuron
    from scalar to vector. So its input shape = [None, input_num_capsule, input_dim_vector] and output shape = \
    [None, num_capsule, dim_vector]. For Dense Layer, input_dim_vector = dim_vector = 1.
    :param num_capsule: number of capsules in this layer
    :param dim_vector: dimension of the output vectors of the capsules in this layer
    :param num_routings: number of iterations for the routing algorithm
    """

    def __init__(self, num_capsule, dim_vector, num_routing=3,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_vector = dim_vector
        self.num_routing = num_routing
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_vector]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_vector = input_shape[2]

        # Transform matrix
        self.W = self.add_weight(
            shape=[self.input_num_capsule, self.num_capsule, self.input_dim_vector, self.dim_vector],
            initializer=self.kernel_initializer,
            name='W')

        # Coupling coefficient. The redundant dimensions are just to facilitate subsequent matrix calculation.
        self.bias = self.add_weight(shape=[1, self.input_num_capsule, self.num_capsule, 1, 1],
                                    initializer=self.bias_initializer,
                                    name='bias',
                                    trainable=False)
        self.built = True

    def call(self, inputs, training=None):
        # inputs.shape=[None, input_num_capsule, input_dim_vector]
        # Expand dims to [None, input_num_capsule, 1, 1, input_dim_vector]
        inputs_expand = K.expand_dims(K.expand_dims(inputs, 2), 2)

        # Replicate num_capsule dimension to prepare being multiplied by W
        # Now it has shape = [None, input_num_capsule, num_capsule, 1, input_dim_vector]
        inputs_tiled = K.tile(inputs_expand, [1, 1, self.num_capsule, 1, 1])

        """
        # Begin: inputs_hat computation V1 ---------------------------------------------------------------------#
        # Compute `inputs * W` by expanding the first dim of W. More time-consuming and need batch_size.
        # w_tiled.shape = [batch_size, input_num_capsule, num_capsule, input_dim_vector, dim_vector]
        w_tiled = K.tile(K.expand_dims(self.W, 0), [self.batch_size, 1, 1, 1, 1])
        # Transformed vectors, inputs_hat.shape = [None, input_num_capsule, num_capsule, 1, dim_vector]
        inputs_hat = K.batch_dot(inputs_tiled, w_tiled, [4, 3])
        # End: inputs_hat computation V1 ---------------------------------------------------------------------#
        """

        # Begin: inputs_hat computation V2 ---------------------------------------------------------------------#
        # Compute `inputs * W` by scanning inputs_tiled on dimension 0. This is faster but requires Tensorflow.
        # inputs_hat.shape = [None, input_num_capsule, num_capsule, 1, dim_vector]
        inputs_hat = tf.scan(lambda ac, x: K.batch_dot(x, self.W, [3, 2]),
                             elems=inputs_tiled,
                             initializer=K.zeros([self.input_num_capsule, self.num_capsule, 1, self.dim_vector]))
        # End: inputs_hat computation V2 ---------------------------------------------------------------------#
        """
        # Begin: routing algorithm V1, dynamic ------------------------------------------------------------#
        def body(i, b, outputs):
            c = tf.nn.softmax(b, dim=2)  # dim=2 is the num_capsule dimension
            outputs = squash(K.sum(c * inputs_hat, 1, keepdims=True))
            if i != 1:
                b = b + K.sum(inputs_hat * outputs, -1, keepdims=True)
            return [i-1, b, outputs]
        cond = lambda i, b, inputs_hat: i > 0
        loop_vars = [K.constant(self.num_routing), self.bias, K.sum(inputs_hat, 1, keepdims=True)]
        shape_invariants = [tf.TensorShape([]),
                            tf.TensorShape([None, self.input_num_capsule, self.num_capsule, 1, 1]),
                            tf.TensorShape([None, 1, self.num_capsule, 1, self.dim_vector])]
        _, _, outputs = tf.while_loop(cond, body, loop_vars, shape_invariants)
        # End: routing by aggrement  algorithm 1, dynamic ------------------------------------------------------------#
        """

        # Begin: routing algorithm V2, static -----------------------------------------------------------#
        # Routing algorithm V2. Use iteration. V2 and V1 both work without much difference on performance
        assert self.num_routing > 0, 'The num_routing should be > 0.'
        for i in range(self.num_routing):
            c = tf.nn.softmax(self.bias, dim=2)
            
            # dim=2 is the num_capsule dimension
            # outputs.shape=[None, 1, num_capsule, 1, dim_vector]
            outputs = squash(K.sum(c * inputs_hat, 1, keepdims=True))

            # last iteration needs not compute bias which will not be passed to the graph any more anyway.
            if i != self.num_routing - 1:
                # self.bias = K.update_add(self.bias, K.sum(inputs_hat * outputs, [0, -1], keepdims=True))
                self.bias += K.sum(inputs_hat * outputs, -1, keepdims=True)
                # tf.summary.histogram('BigBee', self.bias)  # for debugging
        # End: routing algorithm V2, static ------------------------------------------------------------#

        return K.reshape(outputs, [-1, self.num_capsule, self.dim_vector])

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_vector])


# In[14]:


def PrimaryCap(inputs, dim_vector, n_channels, kernel_size, strides, padding):
    """
    Apply Conv1D `n_channels` times and concatenate all capsules
    :param inputs: 4D tensor, shape=[None, width, height, channels]
    :param dim_vector: the dim of the output vector of capsule
    :param n_channels: the number of types of capsules
    :return: output tensor, shape=[None, num_capsule, dim_vector]
    """
    output = layers.Conv1D(filters=dim_vector * n_channels, kernel_size=kernel_size, strides=strides, padding=padding,
                           name='primarycap_conv1d')(inputs)
    outputs = layers.Reshape(target_shape=[-1, dim_vector], name='primarycap_reshape')(output)
    return layers.Lambda(squash, name='primarycap_squash')(outputs)


# In[16]:


def CapsNet(input_shape, n_class, num_routing):

    x = layers.Input(shape=input_shape)
    conv1 = layers.Conv1D(filters=32, kernel_size=7, strides=1, padding='valid', activation='relu', name='conv1')(x)
    conv1=Dropout(0.7)(conv1)
    lstm = layers.LSTM(units=128,   name='Lstm',  return_sequences=True)(conv1)
    # Layer 2: Conv1D layer with `squash` activation, then reshape to [None, num_capsule, dim_vector]
    primarycaps = PrimaryCap(lstm, dim_vector=8, n_channels=16, kernel_size=3, strides=1, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    KcrCaps = CapsuleLayer(num_capsule=n_class, dim_vector=8, num_routing=num_routing, name='KcrCaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out = Length(name='capsnet')(KcrCaps)
    #model
    train_model = Model(x, out)
    return train_model


# In[17]:


def margin_loss(y_true, y_pred):
    """
     When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) +         0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


# ### Our DeepCap-Kcr model

# In[19]:


#31 is length of each sequence and 20 is the one-hot encoding vector for each amino acids, which means we used 31 lenthg of sequence
mode1= CapsNet(input_shape=(31,20),n_class=2,num_routing=3)
mode1.summary()


# ### We Uploaded  and read the independent data set ( as described in Materials and methods (dataset) in the manuscript) 

# In[123]:


# this function is for prediction using the trained five fold models named str (i)Lateopch_new.h5
def get_pred():
    positive_ind_data=read_fasta('pos_ind_testdata.txt') # positive data # use read_rasta function written in the beginning
    print("The total length of positive sequences to be predicted is:", len(positive_ind_data))
    negative_ind_data=read_fasta('neg_ind_testdata.txt') #negative data
    print("The total length of negative sequences to be predicted is:", len(negative_ind_data))
    all_ind_data=positive_ind_data+negative_ind_data
    all_ind_onehot=onehot(all_ind_data) # use onehot function  written in the beginning
    all_pred_y=[]
    for i in range(5):
        mode1.load_weights(str(i+0)+'Lastepoch_new'+'.h5')
        pred_y=mode1.predict(all_ind_onehot)
        #all_pred_y.append(pred_y)
        pred_y=np.argmax(pred_y, axis=1)
        all_pred_y.append(pred_y)
        print(pred_y)
    all_pred_y=np.average(all_pred_y,axis=0)
    print(all_pred_y)
    return all_pred_y


# In[124]:


# result for each fold , 1's is for Kcr sites and 0's is for non-Kcr sites
result=get_pred()


# In[131]:


print("The result shows prediction for first 1000 positive sequence :",' \n', result[:1000]) # prediction of first 10000 positive samples, results shows the accurate prediction 


# In[136]:


print("The result shows prediction for last 1000 negative sequence :",' \n',"jk", result[-1000:]) # prediction of last 10000 negative samples, results shows the accurate prediction 

