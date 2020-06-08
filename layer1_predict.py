from layer1_model import myModel
from utils import predict
import numpy as np


model = myModel(input_shape = (64, 64, 3), classes = 6)

try:
    model.load_weights('layer1_weights.h5')
    print("weights loaded")
except:
    print("weights not found. initiaized randomly")

d = predict(model,'images/edit2.jpg')
idx = np.argmax(d)
print('images is' , idx)
