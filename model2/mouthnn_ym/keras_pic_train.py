import cv2
import os
import numpy as np
from shutil import copyfile

import keras
from keras.preprocessing.image import ImageDataGenerator#, img_to_array, array_to_img
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Activation, Flatten#, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

n_features = 512

def copy_pictures():
    for i in range(2):
        src = '%d/'%i
        for root,subdir,files in os.walk(src):
            X_train, X_test = train_test_split(files, test_size=0.2, 
                                               random_state=1234,
                                               shuffle=True)
            dataset = [X_train, X_test]
            idx = 0
            for sets in ['train', 'test']:
                dst = sets+'/%d/'%i
                if not os.path.exists(dst):
                    os.mkdir(dst)
                for file in dataset[idx]:
                    copyfile(src+file, dst+file)
                idx += 1
                
def dup_pictures():
    for cset in ['train', 'test']:
        dirs = [cset+'/0/', cset+'/1/']
        count = [0, 0]
        for idx, d in enumerate(dirs):
            for root,subdir,files in os.walk(d):
                count[idx] = len(files)
        print(count)
        copy_list = []
        for idx,subdir,files in os.walk(dirs[1]):
            copy_list = files
        mult = int(count[0] / count[1])
        print('copy %d times needed!'%mult)
        for i in range(mult):
            print('copy for %d ...'%i)
            for f in copy_list:
                copyfile(dirs[1]+f, dirs[1]+'%d_'%i+f)
        for idx, d in enumerate(dirs):
            for root,subdir,files in os.walk(d):
                count[idx] = len(files)
        print(count)
    

def ym_preproc(img):
    img = img.astype('float')
    img /= 127.5
    img -= 1.
    return img

def conv_layer_build(tensor, num):
    #tensor = Conv2D(num, kernel_size=(3, 3), strides=(2,2), padding='same')(tensor)
    tensor = Conv2D(num, kernel_size=(3, 3), strides=(2,2))(tensor)
    tensor = BatchNormalization()(tensor)
    tensor = Activation('relu')(tensor)
    return tensor

def custom_model_1st(in_shape, n_features):              
    tin = Input(shape=(in_shape))
    tensor = conv_layer_build(tin, 32)
    tensor = conv_layer_build(tensor, 32)
    tensor = conv_layer_build(tensor, 64)
    tensor = conv_layer_build(tensor, 64)
    tensor = conv_layer_build(tensor, 128)
    #tensor = Flatten()(tensor)
    #fea_out = Dense(n_features)(tensor)
    fea_out = Flatten()(tensor)
    y_pred = Dense(1, activation='sigmoid')(fea_out)
    model = Model(inputs=tin, outputs=y_pred)
    opt = keras.optimizers.Adam()
    model.compile(loss='binary_crossentropy', #loss='mean_squared_error',
                  optimizer=opt,
                  metrics=['accuracy'])
    print(model.summary())
    return model

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

#copy_pictures()
#dup_pictures()


traingen = ImageDataGenerator(
    preprocessing_function = ym_preproc,
    rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
    #zoom_range=[0.8,1.2],
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False,  # randomly flip images
    zca_whitening=False)
            
train_generator = traingen.flow_from_directory(
    directory='./train/',
    target_size=shape_used,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode="binary",
    #class_mode='sparse',
    shuffle=True,
    seed=42)

testgen = ImageDataGenerator(
    preprocessing_function = ym_preproc,
    rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
    #zoom_range=[0.8,1.2],
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False,  # randomly flip images
    zca_whitening=False)

test_generator = testgen.flow_from_directory(
    directory='./test/',
    target_size=shape_used,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode="binary",
    #class_mode='sparse',
    shuffle=True,
    seed=42)

model = custom_model_1st((shape_used[0], shape_used[1], 1), n_features)


checkpoint = ModelCheckpoint('mouthnnym_fea_'+str(n_features)+'.h5', monitor='val_loss',
                             verbose=1, save_best_only=True)
earlystop = EarlyStopping(monitor='val_loss', patience=n_patience, verbose=1)

model.fit_generator(train_generator,
                    epochs=epochs,
                    validation_data=test_generator,
                    workers=4,
                    callbacks=[earlystop,checkpoint])
