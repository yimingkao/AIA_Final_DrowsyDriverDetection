
import cv2
import numpy as np

import keras
from keras.models import load_model
from keras.models import Model

class MouthnnYMFeatureExtract(object):
    # n_features[in] number of features
    def __init__(self, n_features):
        base_model = load_model('mouthnnym_fea_%d.h5'%n_features)
        base_model.layers.pop() # pop last dense(1) layer
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
        img = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        img = img.astype('float')
        img /= 127.5
        img -= 1.
        img = img[:,:,np.newaxis]
        return img
    
    # img[in] opecv BGR frame that contains the face region to do feature extraction
    # pred[out] features extracted
    def feature_extract(self, img):
        img = cv2.resize(img, (100, 100))
        xtest = np.empty(shape=(1, 100, 100, 1))
        xtest[0,:,:,:] = self.preprocess(img)
        pred = self.model.predict(xtest)
        return np.array(pred[0])
