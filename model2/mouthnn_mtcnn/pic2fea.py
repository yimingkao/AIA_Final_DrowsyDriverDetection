import cv2
import os
import numpy as np
import pickle

import keras
from keras.utils.generic_utils import CustomObjectScope
from keras.models import load_model
from keras.models import Model
from keras.applications.mobilenet import relu6
from keras.applications.mobilenet import DepthwiseConv2D

import matplotlib
matplotlib.use('Agg')

def custom_preporc(x):
    x /= 127.5
    x -= 1.
    return x

right_files = []
with open('right.pickle', 'rb') as f:
    right_files = pickle.load(f)

shape_used = (224, 224)

with CustomObjectScope({'relu6': relu6,
                        'DepthwiseConv2D': DepthwiseConv2D}):
    base_model = load_model('mobilecus_fea_512.h5')
    base_model.layers.pop()
    base_model.layers.pop()
    features = base_model.layers[-1].output
    model = Model(input = base_model.input, output = features)
    opt = keras.optimizers.Adam()
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    print(model.summary())

    save = []
    for idx, file in enumerate(right_files):
        img = cv2.imread(file)
        img = cv2.resize(img, shape_used, interpolation=cv2.INTER_AREA)
        img = img[:,:,::-1] # BGR -> RGB
        img = img.astype('float')
        #img = mobilenet_preproc(img)
        img = custom_preporc(img)
        x_test = np.empty(shape=(1, shape_used[0], shape_used[1], 3))
        x_test[0,:,:,:] = img
        y_pred = model.predict(x_test)
        #print(file, y_pred.shape)
        save.append([file, y_pred[0][0][0]])
        print('\r%d'%idx, end='')
        #print(save)
        #break
    with open('right_features.pickle', 'wb') as f:
        pickle.dump(save, f)