from keras.callbacks import Callback

from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.densenet import DenseNet121
from keras.applications.densenet import DenseNet169
from keras.applications.densenet import DenseNet201
#from keras.applications.mobilenet import MobileNet


from keras.applications.vgg16 import preprocess_input as vgg16_preproc
from keras.applications.vgg19 import preprocess_input as vgg19_preproc
from keras.applications.resnet50 import preprocess_input as resnet50_preproc
from keras.applications.inception_v3 import preprocess_input as inception_v3_preproc
from keras.applications.xception import preprocess_input as xception_preproc
from keras.applications.inception_resnet_v2 import preprocess_input as inception_resnet_v2_preproc
from keras.applications.densenet import preprocess_input as densenet_preproc
#from keras.applications.mobilenet import preprocess_input as mobilenet_preproc

import numpy as np
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class YMModelCheckpoint(Callback):
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1, threshold=0, mycb=0):
        super(YMModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.threshold = threshold
        self.mycb = mycb

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('YMModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.threshold):
                        if self.monitor_op(current, self.best):
                            if self.verbose > 0:
                                print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                      ' saving model to %s'
                                      % (epoch + 1, self.monitor, self.best,
                                         current, filepath))
                            self.best = current
                            #if self.save_weights_only:
                            #    self.model.save_weights(filepath, overwrite=True)
                            #elif current < 0.8:
                            #    self.model.save(filepath, overwrite=True)
                            if self.mycb:
                                if self.mycb(current, logs.get('val_acc'), self.model):
                                    #if current < 0.8:
                                    self.model.save(filepath, overwrite=True)
                        else:
                            if self.verbose > 0:
                                print('\nEpoch %05d: %s did not improve from %0.5f' %
                                      (epoch + 1, self.monitor, self.best))
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not pass threshold %0.5f' %
                                 (epoch + 1, self.monitor, self.threshold))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)



def lookup_model(model):
#    if model == 'VGG16':
#        return {
#            'shape': (224, 224),
#            'batch_size': 128,
#            'preproc': vgg16_preproc,
#            'model': VGG16(weights='imagenet', include_top=True,
#                           input_shape=(224, 224, 3)),
#        }
#    elif model == 'VGG19':
#        return {
#            'shape': (224, 224),
#            'batch_size': 128,
#            'preproc': vgg19_preproc,
#            'model': VGG19(weights='imagenet', include_top=True,
#                           input_shape=(224, 224, 3)),
#        }
#    elif model == 'ResNet50':
    if model == 'ResNet50':
        return {
            'shape': (224, 224),
            'batch_size': 64,
            'preproc': resnet50_preproc,
            'model': ResNet50(weights='imagenet', include_top=True,
                         input_shape=(224, 224, 3)),
        }
#    elif model == 'inception_v3':
#        return {
#            'shape': (299, 299),
#            'batch_size': 32,
#            'preproc': inception_v3_preproc,
#            'model': InceptionV3(weights='imagenet', include_top=True,
#                         input_shape=(299, 299, 3)),
#        }
    elif model == 'xception':
        return {
            'shape': (299,299),
            'batch_size': 16,
            'preproc': xception_preproc,
            'model': Xception(weights='imagenet', include_top=True,
                         input_shape=(299, 299, 3)),   
        }
#    elif model == 'inception_resnet_v2':
#        return {
#            'shape': (299, 299),
#            'batch_size': 32,
#            'preproc': inception_resnet_v2_preproc,
#            'model': InceptionResNetV2(weights='imagenet', include_top=True,
#                         input_shape=(299, 299, 3)),            
#        }
#    elif model == 'densenet121':
#        return {
#            'shape': (224, 224),
#            'batch_size': 32,
#            'preproc': densenet_preproc,
#            'model': DenseNet121(weights='imagenet', include_top=True,
#                         input_shape=(224, 224, 3)),
#        }
#    elif model == 'densenet169':
#        return {
#            'shape': (224, 224),
#            'batch_size': 32,
#            'preproc': densenet_preproc,
#            'model': DenseNet169(weights='imagenet', include_top=True,
#                         input_shape=(224, 224, 3)),
#        }
#    elif model == 'densenet201':
#        return {
#            'shape': (224, 224),
#            'batch_size': 32,
#            'preproc': densenet_preproc,
#            'model': DenseNet201(weights='imagenet', include_top=True,
#                         input_shape=(224, 224, 3)),
#        }
#    elif model == 'mobilenet':
#        return {
#            'shape': (224, 224),
#            'batch_size': 128,
#            'preproc': mobilenet_preproc,
#            'model': MobileNet(alpha=1.0, depth_multiplier=1, dropout=1e-3, 
#                               weights='imagenet', input_shape=(224, 224, 3)),            
#        }
    else:
        return {}

def pop_layer(model):
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')

    model.layers.pop()
    if not model.layers:
        model.outputs = []
        model.inbound_nodes = []
        model.outbound_nodes = []
    else:
        model.layers[-1].outbound_nodes = []
        model.outputs = [model.layers[-1].output]
    model.built = False
    
    
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
    plt.clf()