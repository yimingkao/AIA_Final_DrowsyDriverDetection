import cv2
import os
import numpy as np

import keras
from keras.preprocessing.image import ImageDataGenerator#, img_to_array, array_to_img
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Activation, Flatten#, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

n_features = 512

def ym_preproc(img):
    img = img.astype('float')
    img /= 127.5
    img -= 1.
    return img

def load_set(path, shape, value):
    for root,subdir,files in os.walk(path):
        file_count = len(files)
        X = np.empty(shape=(file_count, shape[0], shape[1], 1))
        y = np.empty(shape=(file_count))
        for idx, file in enumerate(files):
            img = cv2.imread(path+file)
            img = cv2.resize(img, shape, interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = ym_preproc(img)
            img = img[:,:,np.newaxis]
            X[idx] = img
            y[idx] = value
    return X, y

def custom_model(in_shape, n_features):              
    model = load_model('mouthnnym_fea_'+str(n_features)+'.h5')
    opt = keras.optimizers.Adam(lr=1e-4)
    model.compile(loss='binary_crossentropy', #loss='mean_squared_error',
                  optimizer=opt,
                  metrics=['accuracy'])
    print(model.summary())
    return model

shape_used = (100, 100)
epochs = 300
batch_size = 128
n_patience = 30

model = custom_model((shape_used[0], shape_used[1], 1), n_features)
X_test, y_true = load_set('0/', shape_used, 0)
y_pred = model.predict(X_test)
print(y_pred.shape)
count = 0
for i in range(len(y_true)):
    if y_true[i] != int(y_pred[i]+0.499):
        count+=1
print('0: %d, %f'%(count, (len(y_true) - count)/len(y_true)))

X_test, y_true = load_set('1/', shape_used, 1)
y_pred = model.predict(X_test)
count = 0
print(y_pred.shape)
for i in range(len(y_true)):
    if y_true[i] != int(y_pred[i]+0.499):
        count+=1
print('1: %d, %f'%(count, (len(y_true) - count)/len(y_true)))