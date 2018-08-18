
import cv2
import numpy as np
import keras
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope
from keras.applications.mobilenet import preprocess_input as mobilenet_preproc
# Server
from keras.applications.mobilenet import relu6
from keras.applications.mobilenet import DepthwiseConv2D
# PC (Keras2.2)
#from keras_applications.mobilenet import relu6
#from keras.layers import DepthwiseConv2D
from keras.models import Model
from keras.layers import Dense


class MobileNetCustomFeatureExtract(object):
    # n_features[in] number of features
    def __init__(self, n_features):
        with CustomObjectScope({'relu6': relu6,
                                'DepthwiseConv2D': DepthwiseConv2D}):
            base_model = load_model('mobilecus_fea_%d.h5'%n_features)
            base_model.layers.pop()
            base_model.layers.pop()
            features = base_model.layers[-1].output
            self.model = Model(input = base_model.input, output = features)
            # We need optimizer even we just want to extract feature.
            opt = keras.optimizers.Adam()
            self.model.compile(loss='categorical_crossentropy',
                              optimizer=opt,
                              metrics=['accuracy'])
            print(self.model.summary())
    
    # x[in] opencv BGR frame.
    # img[out] preprocessed by dense121 model
    def preprocess(self, x):
        img = x[:,:,::-1] # opencv BGR -> RGB
        img = mobilenet_preproc(img.astype(float))
        return img
    
    # img[in] opecv BGR frame that contains the face region to do feature extraction
    # pred[out] features extracted
    def feature_extract(self, img):
        img = cv2.resize(img, (224, 224))
        xtest = np.empty(shape=(1, 224, 224, 3))
        xtest[0,:,:,:] = self.preprocess(img)
        pred = self.model.predict(xtest)
        return pred
