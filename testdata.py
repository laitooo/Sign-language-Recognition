import pylab
import imageio
import numpy as np
from utils import *
from layer1_model import myModel



X,Y,xt,yt = load_dataset()



model = myModel(input_shape = (64, 64), classes = 28)
model.load_weights('layer1_weights.h5')

sl = [2,66,736,1356,16372,7654,476,43,7]

for i,im in enumerate(sl):
    save_image(X[im], 'tests/image' + str(i) + '.jpg')
    x = X[im].reshape(1,64,64)
    x = x/255.
    res = model.predict(x)
    idx = np.argmax(res)
    print('image no.',i ,' is actually a ',Y[im],' but predicted as a ',idx)
    
