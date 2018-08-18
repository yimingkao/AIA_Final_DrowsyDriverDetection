
import cv2
import numpy as np
import keras
from keras.applications.mobilenetv2 import MobileNetV2
from keras.applications.mobilenetv2 import preprocess_input as preproc

from keras.models import Model
from keras.layers import Dense


class MobileNetV2FeatureExtract(object):
    # n_features[in] number of features
    def __init__(self, n_features):
        #self.shape = (224, 224)
        #self.n_features = n_features
        base_model = MobileNetV2(weights='imagenet', include_top=True, 
                         input_shape=(224, 224, 3))
        base_model.layers.pop()
        x = base_model.layers[-1].output
        features = Dense(n_features)(x)
        self.model = Model(input = base_model.input, output = features)
        # We need optimizer even we just want to extract feature.
        opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None)
        self.model.compile(loss='categorical_crossentropy',
                          optimizer=opt,
                          metrics=['accuracy'])
        print(self.model.summary())
    
    # x[in] opencv BGR frame.
    # img[out] preprocessed by dense121 model
    def preprocess(self, x):
        img = x[:,:,::-1] # opencv BGR -> RGB
        img = preproc(img.astype(float))
        return img
    
    # img[in] opecv BGR frame that contains the face region to do feature extraction
    # pred[out] features extracted
    def feature_extract(self, img):
        img = cv2.resize(img, (224, 224))
        xtest = np.empty(shape=(1, 224, 224, 3))
        xtest[0,:,:,:] = self.preprocess(img)
        pred = self.model.predict(xtest)
        return pred
