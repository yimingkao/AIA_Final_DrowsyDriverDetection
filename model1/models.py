import keras
from keras import backend
#from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping
#from keras.callbacks import LearningRateScheduler

#from sklearn.model_selection import train_test_split

#import numpy as np
#import pandas as pd
import sys
import os
import gc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ymutils import lookup_model
from ymutils import YMModelCheckpoint

### Helper function ####
def YMCB(cur_loss, cur_acc, model):
    global base_loss
    global best_loss
    global base_acc
    global best_acc
    global model_path
    if base_acc < cur_acc:
        base_acc = cur_acc
    if best_acc < cur_acc:
        best_acc = cur_acc
        print('\nUpdate BEST ACC %.4f'%best_acc)
        if cur_acc > 0.9: # save model
            model.save(model_path+'.acc', overwrite=True)
    if cur_loss < base_loss:
        print('\nUpdate loss from %.4f to %.4f'%(base_loss, cur_loss))
        base_loss = cur_loss
    if cur_loss < best_loss:
        best_loss = cur_loss
        print('\nUpdate BEST LOSS %.4f'%best_loss)
        return True
    return False
###
    
def preprocess_input(x):
    arr = model_dict['preproc'](img_to_array(x))
    return array_to_img(arr)

hist_lr = []
lr_loss = 10.
lr_patience = 0
base_loss = 10.
base_acc = 0
best_loss = 10.
best_acc = 0
model_path = ''
opt = ''
model_dict = None
def train_with_model(model_used, nth, nth_loss, nth_acc):
    global best_loss
    global best_acc
    global model_path
    global opt
    global model_dict
    best_loss = nth_loss
    best_acc = nth_acc
    # model parameters
    #model_used = 'ResNet50'
    model_dict = lookup_model(model_used)
    fixed_used   = False
    flatten_used = False
    avgpool_used = False
    dropout_used = 0#0.25

    #train parameters
    num_classes = 6
    epochs = 100
    n_patience = 10
    model_name = model_used+'_trained.h5'

    shape_used = model_dict['shape']
    batch_size = model_dict['batch_size']

    save_dir = os.path.join(os.getcwd(), 'saved_models')
    # Use ModelCheckpoint to save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)

    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    traingen = ImageDataGenerator(
        preprocessing_function = preprocess_input,
        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        zoom_range=[0.8,1.2],
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        zca_whitening=False)
    
    train_generator = traingen.flow_from_directory(
        directory='train/',
        target_size=shape_used,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=42)
    
    validgen = ImageDataGenerator(
        preprocessing_function = preprocess_input,
        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        zoom_range=[0.8,1.2],
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        zca_whitening=False)
    
    valid_generator = validgen.flow_from_directory(
        directory='valid/',
        target_size=shape_used,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=42)


    base_model = model_dict['model']
    #print(base_model.summary())
    base_model.layers.pop()
    x = base_model.layers[-1].output
    if fixed_used:
        for layer in base_model.layers:
            layer.trainable = False
    if flatten_used:
        x = Flatten()(base_model.output)
    elif avgpool_used:
        x = GlobalAveragePooling2D()(base_model.output)
    if dropout_used > 0.001:
        x = Dropout(dropout_used)(x)
    predictions = Dense(num_classes, activation = 'softmax')(x)

    #create graph of your new model
    model = Model(input = base_model.input, output = predictions)
    print(model.summary())
    print('(%d) best loss/acc: %.5f, %.5f'%(nth, best_loss, best_acc))

    # initiate Adam optimizer
    opt = keras.optimizers.Adam(lr=0.0001/10, beta_1=0.9, beta_2=0.999, epsilon=None)
    #opt = keras.optimizers.SGD(lr=0.0001/10, decay=1e-6, momentum=0.9, nesterov=True)

    # Let's train the model using Adam
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    checkpoint = YMModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, 
                                   verbose=0, threshold=10., mycb=YMCB)
    
#    lr_modifier = LearningRateScheduler(lrScheduler)

    # earlystop
    earlystop = EarlyStopping(monitor='val_loss', patience=n_patience, verbose=1)

    model_history = model.fit_generator(train_generator,
                        epochs=epochs,
                        #validation_data=(x_valid, y_valid),
                        validation_data=valid_generator,
                        workers=4,
                        callbacks=[earlystop,checkpoint])
    del model
    backend.clear_session()
    gc.collect()
    
    # Output log to file
    with open(model_used + ".log", "w") as f:
        string = '%d,%.5f,%.5f'%(nth, best_loss, best_acc)
        f.write(string)
    
    #print(hist_lr)

    # Output Model information
    print(model_used + ' best loss %.4f, best acc %.4f' % (best_loss, best_acc));
    fname = model_used + '_%d_loss_%.3f_acc_%.3f' % (nth, base_loss, base_acc)
    training_loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']
    #loss graph
    plt.subplot(121)
    plt.plot(training_loss, label="training_loss")
    plt.plot(val_loss, label="validation_loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.legend(loc='best')
    plt.tight_layout()
    #Accuracy graph
    plt.subplot(122)
    plt.plot(model_history.history['acc'], label="training_acc")
    plt.plot(model_history.history['val_acc'], label="validation_acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(fname+'.png')
    plt.clf()
    #plt.show()
    
    
def main():
    if len(sys.argv) < 2:
        print('Usage: ' + sys.argv[0] + ' model')
        print('Support models: ResNet50, inception_v3, xception, inception_resnet_v2, densenet201')
        sys.exit(1)
    model = sys.argv[1]
    if len(sys.argv) >= 5:
        nth = int(sys.argv[2])
        best_loss = float(sys.argv[3])
        best_acc = float(sys.argv[4])
    else:
        with open(model+'.log', 'r') as f:
            line = f.readline()
        nth, best_loss, best_acc = line.split(',')
        nth = int(nth) + 1
        best_loss = float(best_loss)
        best_acc = float(best_acc)
    print(nth, best_loss, best_acc)
    train_with_model(model, nth, best_loss, best_acc)

if __name__ == "__main__":
    main()
