#######################################################################################################

print("Arabic Sign language detection project")

# Importing the modules

import numpy as np
from keras import layers
import keras.layers as kl
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
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
from utils import *
from layer1_model import myModel
from myCallback import *


print("Modules loaded")

#######################################################################################################

# Loading the data (signs)

X_train, Y_train, X_test, Y_test = load_dataset()

X_train = X_train/255.
X_test = X_test/255.
Y_train = convert_to_one_hot(Y_train, 28).T
Y_test = convert_to_one_hot(Y_test, 28).T

m_train = X_train.shape[0]
m_test = X_test.shape[0]
print()
print ("number of training examples = " + str(m_train))
print ("number of test examples = " + str(m_test))

X_train = X_train.reshape(m_train,64,64,1)
X_test = X_test.reshape(m_test,64,64,1)

print()
print("Dataset loaded")
print()
print()


#######################################################################################################

# Loading the model

model = myModel(input_shape = (64, 64, 1), classes = 28)

try:
    model.load_weights('layer1_weights.h5')
    print("weights loaded")
except:
    print("weights not found. initiaized randomly")

print("model ready")
print()
print()

###################################################################################################


# Model trainning

print("trainning started")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs = 20, batch_size = 128, verbose = 0, callbacks=[MyCustomCallback()])

print()
print("model trained")

###################################################################################################


# Model evaluating

print()
#model.summary()
print("Evaluating...")

print()
preds = model.evaluate(X_test, Y_test, verbose=0)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

###################################################################################################


# Model saving

print("previous test accuracy : 0.0865476205945015   loss: 5.757261755807059 ")
print("trained for 40 ebochs")
model.save_weights('layer1_weights.h5')
print("weights saved")


##################################################################################################

