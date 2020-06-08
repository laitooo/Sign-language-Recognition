import pylab
import imageio
import numpy as np
from utils import *
from layer1_model import myModel

letters = ['أ','ب','ت','ث','ج','ح','خ','د','ذ','ر','ز','س','ش','ص','ض','ط',
           'ظ','ع','غ','ف','ق','ك','ل','م','ن','ه','و','ي']


filename = 'videos/video5.mp4'
vid = imageio.get_reader(filename,  'ffmpeg')
count = 0
for _ in vid:
    count = count +1
video = np.zeros((count,640,640,3))

for i, im in enumerate(vid):
    video[i] = np.array(im, dtype='int64')




model = myModel(input_shape = (64, 64), classes = 28)
model.load_weights('layer1_weights.h5')


def predict_frame(vid,i):
    img = video[i,:,:,:]
#save_image(img, 'images/image1.jpg')
#showImage(img)
#gray = rgb_to_gray(img)
#save_image(gray, 'images/image2.jpg')
#showImage(gray)
    tmp = np.array(resizeImage(img))
    tmp.reshape(64,64,3)
#save_image(tmp, 'images/image3.jpg')
    x = np.array(rgb_to_gray(tmp))
#save_image(x, 'images/image4.jpg')
    save_image(x, 'images/image' + str(i) + '.jpg')
    x = x.reshape(1,64,64)
    x = x/255.
    #print(x.shape)


    res = model.predict(x)
    #idx = np.argmax(res)
    #print('frame : ', i, ' index : ' , idx ,' result is : ', letters[idx])
    #return idx
    return res



l = []
le = []
threshold = 0.998
for (i,im) in enumerate(vid):
    res = predict_frame(vid,i)
    tmp = res[res>threshold]
    if (tmp.shape[0] > 0):
        t = np.argmax(res)
        l.append(t)
        le.append(letters[t])
    

threshold = 4
c = 0
s = 0
m = []
for i in l:
    if (i == s):
        c = c+1
        if (c == threshold):
            m.append(letters[s])
            c = 0
    else :
        c = 1
        s = i

print(l,end='\n\n\n')
print(le,end='\n\n\n')
print(m,end='\n\n\n')


print('numper of frames : ' + str(count))
print('number of accurates : ' + str(len(l)))
