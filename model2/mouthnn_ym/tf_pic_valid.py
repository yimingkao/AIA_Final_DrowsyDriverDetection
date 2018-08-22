
import os
import cv2
import numpy as np
import tensorflow as tf

def conv_layer_build(x_tensor, num, is_training, scope):
    x1 = tf.layers.conv2d(x_tensor, num, kernel_size=(3,3), strides=(2,2),
                          name=scope)
    x1 = tf.contrib.layers.batch_norm(x1, 
                                      center=True, scale=True, 
                                      is_training=is_training,
                                      scope=scope+'_bn')
    x1 = tf.nn.relu(x1, name=scope+'_relu')
    return x1

# x[in] opencv BGR frame.
# img[out] preprocessed by dense121 model
def preprocess(x):
    img = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    img = img.astype('float')
    img /= 127.5
    img -= 1.
    img = img[:,:,np.newaxis]
    return img

n_features = 512
shape_used = (100, 100)
tf.reset_default_graph()
is_training = tf.placeholder(dtype=tf.bool)
input_data = tf.placeholder(dtype=tf.float32, 
                           shape=[None, shape_used[0], shape_used[1], 1],
                           name='input_data')
#y_true = tf.placeholder(dtype=tf.float32, 
#                        shape=[None, n_features],
#                        name='y_true')

x1 = conv_layer_build(input_data, 32, is_training, 'conv1')
x2 = conv_layer_build(x1, 32, is_training, 'conv2')
x3 = conv_layer_build(x2, 64, is_training, 'conv3')
x4 = conv_layer_build(x3, 64, is_training, 'conv4')
x5 = conv_layer_build(x4, 128, is_training, 'conv5')
flatten = tf.layers.flatten(x5)
fea_out = tf.layers.dense(flatten, n_features, name='fea_out')# output layer
y_pred = tf.layers.dense(fea_out, 1, name='output') # output layer
saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, './mouthnn_ym.ckpt')

total = 0
for i in range(6):
    src = 'test/'+str(i)
    for root,subdir,files in os.walk(src):
        total += len(files)
print(total)
X_test = np.empty(shape=(total, shape_used[0], shape_used[1], 1))
y_test = np.empty(shape=(total))
idx = 0
for i in range(6):
    src = 'test/'+str(i)
    for root,subdir,files in os.walk(src):
        for j, file in enumerate(files):
            img = cv2.imread(root+'/'+file)
            img = cv2.resize(img, shape_used, interpolation=cv2.INTER_AREA)
            img = preprocess(img)
            X_test[idx+j] = img
            y_test[idx+j] = i
        idx += len(files)

pred = sess.run([y_pred], feed_dict={
    input_data: X_test,
    is_training: False
})
pred = np.array(pred)
print(np.max(pred), np.min(pred))
