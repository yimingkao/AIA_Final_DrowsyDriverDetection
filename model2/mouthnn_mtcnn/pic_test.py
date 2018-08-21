import cv2
import os
import numpy as np
import pickle

from keras.utils.generic_utils import CustomObjectScope
from keras.models import load_model
from keras.applications.mobilenet import relu6
from keras.applications.mobilenet import DepthwiseConv2D
from keras.applications.mobilenet import preprocess_input as mobilenet_preproc
import itertools
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def custom_preporc(x):
    x /= 127.5
    x -= 1.
    return x


def plot_confusion_matrix(cm, classes, filename,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(filename)
    plt.close()

shape_used = (224, 224)

with CustomObjectScope({'relu6': relu6,
                        'DepthwiseConv2D': DepthwiseConv2D}):
    model = load_model('mobilecus_fea_512.h5')
    print(model.summary())

    gen_files = []
    for folder in ['test/', 'train/']:
        golden = []
        pred = []
        for i in range(6):
            working_dir = folder+str(i)+'/'
            for root,subdir,files in os.walk(working_dir):
                print('Processing: '+working_dir)
                for file in files:
                    img = cv2.imread(root+file)
                    img = cv2.resize(img, shape_used, interpolation=cv2.INTER_AREA)
                    img = img[:,:,::-1] # BGR -> RGB
                    img = img.astype('float')
                    #img = mobilenet_preproc(img)
                    img = custom_preporc(img)
                    x_test = np.empty(shape=(1, shape_used[0], shape_used[1], 3))
                    x_test[0,:,:,:] = img
                    y_pred = model.predict(x_test)
                    y_pred = int(y_pred[0][0]+0.4)
                    golden.append(i)
                    pred.append(y_pred)
                    if y_pred == i:
                        gen_files.append(root+file)
            #print(max(pred))
        cm = confusion_matrix(golden, pred, labels=[0,1,2,3,4,5])
        plot_confusion_matrix(cm, [0,1,2,3,4,5], folder.replace('/', '.png'))
        correct = 0
        for i in range(len(golden)):
            if golden[i] == pred[i]:
                correct += 1
        print(folder + ': %.4f'%(correct * 100 / len(golden)))
    print(len(gen_files))
    with open('right.pickle', 'wb') as f:
        pickle.dump(gen_files, f)    