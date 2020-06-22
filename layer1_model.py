import numpy as np
from keras import layers
import keras.layers as kl
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Reshape
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
import keras.backend as K





def myModel(input_shape = (64, 64, 1), classes = 28):
    
    X_input = Input(input_shape)

    X = ZeroPadding2D((3,3))(X_input)
    
    # Layer_group_1
    X = Conv2D(classes, kernel_size = (5,5), strides = (3,3), name = 'conv1', padding='same', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(3, strides= (3,3))(X)
    print(X.shape)
    #X = Reshape((128,64,1))(X)

    # Layer_group_2
    X = ZeroPadding2D((3,3))(X)
    X = Conv2D(classes, kernel_size = (5,5), strides = (3,3), name = 'conv2', kernel_initializer = glorot_uniform(seed = 1))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv2')(X)
    #X = Activation('relu')(X)
    #X = MaxPooling1D(2, strides = 2)(X)
    X = AveragePooling2D(3,3)(X)
    X = Activation('sigmoid')(X)
    print(X.shape)
    #16*64 = 32*32
    #X = Reshape((98,64,1))(X)
    X = Conv2D(classes, kernel_size = (5,5), strides = (3,3), name = 'conv3', padding='same', kernel_initializer = glorot_uniform(seed=1))(X)    
    X = BatchNormalization(axis = 3, name = 'bn_conv3')(X)
    X = Activation('relu')(X)
    X = ZeroPadding2D((3,3))(X)
    X = Conv2D(classes, kernel_size = (5,5), strides = (3,3), name = 'conv4', padding='same', kernel_initializer = glorot_uniform(seed = 1))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv4')(X)
    #X = Activation('relu')(X)
    #X = MaxPooling1D(2, strides = 2)(X)
    X = AveragePooling2D(3, 3)(X)
    X = Activation('sigmoid')(X)
    #print(X.shape)
    # Output


    
    X = Flatten()(X)
    #X = Dense(96, activation='softmax', name='fc' + str(1), kernel_initializer = glorot_uniform(seed=0))(X)
    #X = Dense(32, activation='softmax', name='fc' + str(2), kernel_initializer = glorot_uniform(seed=1))(X)
    #X = Dense(32, activation='softmax', name='fc' + str(3), kernel_initializer = glorot_uniform(seed=2))(X)
    X = Dense(classes, activation='softmax', name='fc' + str(4), kernel_initializer = glorot_uniform(seed=4))(X)
    
    model = Model(inputs = X_input, outputs = X, name='myModel')

    return model
