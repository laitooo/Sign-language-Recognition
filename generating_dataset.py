import numpy as np
import h5py
import os
from utils import loadAllImages
from sklearn.model_selection import train_test_split

X = np.zeros((28000,64,64))
Y = np.zeros((28000,1))

for i in range(0,27):
    bn = i*1000
    bm = (i+1)*1000
    X[bn:bm,:,:],Y[bn:bm] = loadAllImages('dataset/',i)
    print(str(i) + ' done from 28')

'''X[0:2000,:,:],Y[0:2000] = loadAllImages('dataset/',0)
X[2000:4000,:,:],Y[2000:4000] = loadAllImages('dataset/',1)
X[4000:6000,:,:],Y[4000:6000] = loadAllImages('dataset/',2)
X[6000:8000,:,:],Y[6000:8000] = loadAllImages('dataset/',3)
X[8000:10000,:,:],Y[8000:10000] = loadAllImages('dataset/',4)
X[10000:12000,:,:],Y[10000:12000] = loadAllImages('dataset/',5)
X[12000:14000,:,:],Y[12000:14000] = loadAllImages('dataset/',6)
X[14000:16000,:,:],Y[14000:16000] = loadAllImages('dataset/',7)
X[16000:18000,:,:],Y[16000:18000] = loadAllImages('dataset/',8)
X[18000:20000,:,:],Y[18000:20000] = loadAllImages('dataset/',9)
X[20000:22000,:,:],Y[20000:22000] = loadAllImages('dataset/',10)
X[22000:24000,:,:],Y[22000:24000] = loadAllImages('dataset/',11)
X[24000:26000,:,:],Y[24000:26000] = loadAllImages('dataset/',12)
X[26000:28000,:,:],Y[26000:28000] = loadAllImages('dataset/',13)
X[28000:30000,:,:],Y[28000:30000] = loadAllImages('dataset/',14)
X[30000:32000,:,:],Y[30000:32000] = loadAllImages('dataset/',15)
X[32000:34000,:,:],Y[32000:34000] = loadAllImages('dataset/',16)
X[34000:36000,:,:],Y[34000:36000] = loadAllImages('dataset/',17)
X[36000:38000,:,:],Y[36000:38000] = loadAllImages('dataset/',18)
X[38000:40000,:,:],Y[38000:40000] = loadAllImages('dataset/',19)
X[40000:42000,:,:],Y[40000:42000] = loadAllImages('dataset/',20)
X[42000:44000,:,:],Y[42000:44000] = loadAllImages('dataset/',21)
X[44000:46000,:,:],Y[44000:46000] = loadAllImages('dataset/',22)
X[46000:48000,:,:],Y[46000:48000] = loadAllImages('dataset/',23)
X[48000:50000,:,:],Y[48000:50000] = loadAllImages('dataset/',24)
X[50000:52000,:,:],Y[50000:52000] = loadAllImages('dataset/',25)
X[52000:54000,:,:],Y[52000:54000] = loadAllImages('dataset/',26)
X[54000:56000,:,:],Y[54000:56000] = loadAllImages('dataset/',31)'''
X[27000:28000,:,:],_ = loadAllImages('dataset/',31)
Y[27000:28000] = 27


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

hf = h5py.File('arabic_signs.h5', 'w')

hf.create_dataset('X_train', data=X_train)
hf.create_dataset('Y_train', data=Y_train)
hf.create_dataset('X_test', data=X_test)
hf.create_dataset('Y_test', data=Y_test)

hf.close()

