import cv2
import os
from shutil import copyfile
import keras
from keras.preprocessing.image import ImageDataGenerator#, img_to_array, array_to_img
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten#, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input as mobilenet_preproc


def preprocess_input(x):
    img = x[:,:,::-1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def copy_pictures():
    for i in range(6):
        src = 'deg%d/'%i
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

def custom_model():              
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(shape_used[0], shape_used[1], 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dense(512, activation='softmax'))
    model.add(Flatten())
    model.add(Dense(6, activation='softmax'))
    opt = keras.optimizers.Adam()
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    print(model.summary())
    return model

def mobilenet_model():
    base_model = MobileNet(weights='imagenet', include_top=True, 
                     input_shape=(224, 224, 3))
    base_model.layers.pop()
    x = base_model.layers[-1].output
    x = Dense(512)(x)
    x = Flatten()(x)
    #n_classes = Dense(6)(x)
    n_classes = Dense(1)(x)
    model = Model(input = base_model.input, output = n_classes)
    #opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None)
    opt = keras.optimizers.Adam()
    model.compile(
        #loss='categorical_crossentropy',
        loss='mean_squared_error',
        optimizer=opt,
        metrics=['accuracy'])
    print(model.summary())
    return model




copy_pictures()
#shape_used = (100, 100)
shape_used = (224, 224)
epochs = 300
batch_size = 32
n_patience = 30

traingen = ImageDataGenerator(
    preprocessing_function = mobilenet_preproc,
    rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
    #zoom_range=[0.8,1.2],
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False,  # randomly flip images
    zca_whitening=False)
            
train_generator = traingen.flow_from_directory(
    directory='train/',
    target_size=shape_used,
    batch_size=batch_size,
    #class_mode="categorical",
    class_mode='sparse',
    shuffle=True,
    seed=42)

testgen = ImageDataGenerator(
    preprocessing_function = mobilenet_preproc,
    rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
    #zoom_range=[0.8,1.2],
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False,  # randomly flip images
    zca_whitening=False)

test_generator = testgen.flow_from_directory(
    directory='test/',
    target_size=shape_used,
    batch_size=batch_size,
    #class_mode="categorical",
    class_mode='sparse',
    shuffle=True,
    seed=42)

#model = custom_model()
model = mobilenet_model()

checkpoint = ModelCheckpoint('custom.h5', monitor='val_loss',
                             verbose=1, save_best_only=True)
earlystop = EarlyStopping(monitor='val_loss', patience=n_patience, verbose=1)

model.fit_generator(train_generator,
                    epochs=epochs,
                    validation_data=test_generator,
                    workers=4,
                    callbacks=[earlystop,checkpoint])
