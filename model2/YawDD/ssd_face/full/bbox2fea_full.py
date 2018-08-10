
import cv2
import pandas as pd
import numpy as np
import re
import os

import keras
from keras.applications.densenet import DenseNet121
from keras.applications.densenet import preprocess_input as densenet_preproc
from keras.models import Model
from keras.layers import Dense

def isMale(name):
    return re.match('[0-9]{1,2}-Male.+', name) != None

# preprocess & convert to RGB arrangement.
def preprocess_input(x):
    img = x[:,:,::-1] # opencv BGR -> RGB
    img = densenet_preproc(img.astype(float))
    return img

video_path = '/projectdata/driver/YawDD/'

N_FEATURES = 2048
shape_used=(224, 224)
base_model = DenseNet121(weights='imagenet', include_top=True, 
                         input_shape=(shape_used[0], shape_used[1], 3))
base_model.layers.pop()
x = base_model.layers[-1].output
features = Dense(N_FEATURES)(x)
model = Model(input = base_model.input, output = features)
# We need optimizer even we just want to extract feature.
opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None)
model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
print(model.summary())

prefix = 'dense121_'+str(N_FEATURES)+'_'

for patterns in ['yawn_train', 'yawn_valid', 'yawn_test']: #要產生的 dataset

    data = pd.read_csv('../../'+patterns+'.csv')
    for i in range(len(data)):
        dstname = prefix+patterns+'/'+data['Name'][i].replace('.avi', '.npy')
        bboxname = '../bbox/'+patterns+'/'+data['Name'][i].replace('.avi', '.csv')
        if os.path.exists(dstname):
            continue
        src_path = video_path
        if isMale(data['Name'][i]):
            src_path += 'Male/'
        else:
            src_path += 'Female/'
        srcname = src_path + data['Name'][i]
        if not os.path.exists(srcname):
            continue
        txtname = srcname.replace('.avi', '_mark.txt')
        vin = cv2.VideoCapture(srcname)
        length = int(vin.get(cv2.CAP_PROP_FRAME_COUNT))
        print('{}: {}'.format(srcname, length))
        fmark = open(txtname, 'r')
        degrees = []
        for j in range(length):
            degree = fmark.readline()
            degrees.append(int(degree))
        fmark.close()
        cord = pd.read_csv(bboxname)    
        fea = np.empty(shape=(length,N_FEATURES+1))
        for j in range(length):
            ret, frame = vin.read()
            startX = cord['sx'][j] if cord['sx'][j] >= 0 else 0
            startY = cord['sy'][j] if cord['sy'][j] >= 0 else 0
            endX = cord['ex'][j]
            endY = cord['ey'][j]
            # Extract features by good model
            face_img = frame[startY:endY, startX:endX]
            face_img = cv2.resize(face_img, shape_used)
            x_test = np.empty(shape=(1, shape_used[0], shape_used[1], 3))
            x_test[0,:,:,:] = preprocess_input(face_img)
            pred = model.predict(x_test)
            pred = np.append(pred, degrees[j])
            fea[j,:] = pred
            print('\r%d'%j, end='')
        vin.release()
        np.save(dstname, fea)
        print('\n')
        #break