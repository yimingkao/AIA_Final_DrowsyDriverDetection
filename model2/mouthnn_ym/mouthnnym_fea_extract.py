
import cv2
import numpy as np
import tensorflow as tf

class MouthnnYMFeatureExtract(object):
    # n_features[in] number of features
    def __init__(self, n_features):
        def conv_layer_build(x_tensor, num, is_training, scope):
            x1 = tf.layers.conv2d(x_tensor, num, kernel_size=(3,3), strides=(2,2),
                                  name=scope)
            x1 = tf.contrib.layers.batch_norm(x1, 
                                              center=True, scale=True, 
                                              is_training=is_training,
                                              scope=scope+'_bn')
            x1 = tf.nn.relu(x1, name=scope+'_relu')
            return x1
        shape_used = (100, 100)
        tf.reset_default_graph()
        self.is_training = tf.placeholder(dtype=tf.bool)
        self.input_data = tf.placeholder(dtype=tf.float32, 
                                   shape=[None, shape_used[0], shape_used[1], 1],
                                   name='input_data')
        #y_true = tf.placeholder(dtype=tf.float32, 
        #                        shape=[None, n_features],
        #                        name='y_true')

        x1 = conv_layer_build(self.input_data, 32, self.is_training, 'conv1')
        x2 = conv_layer_build(x1, 32, self.is_training, 'conv2')
        x3 = conv_layer_build(x2, 64, self.is_training, 'conv3')
        x4 = conv_layer_build(x3, 64, self.is_training, 'conv4')
        x5 = conv_layer_build(x4, 128, self.is_training, 'conv5')
        flatten = tf.layers.flatten(x5)
        self.y_pred = tf.layers.dense(flatten, n_features, name='output')# output layer
        saver = tf.train.Saver()
        self.sess = tf.Session()
        saver.restore(self.sess, 'mouthnnym_saved/mouthnn_ym.ckpt')

    
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
        pred = self.sess.run([self.y_pred], feed_dict={
            self.input_data: xtest,
            self.is_training: False
        })
        #print(np.array(pred[0][0]).shape)
        return np.array(pred[0][0])
