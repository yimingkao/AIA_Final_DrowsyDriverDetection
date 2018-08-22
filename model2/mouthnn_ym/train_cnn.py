import cv2
import pickle
import numpy as np
from sklearn.utils import shuffle 
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

def conv_layer_build(x_tensor, num, scope):
    x1 = tf.layers.conv2d(x_tensor, 32, kernel_size=(3,3), strides=(2,2),
                          name=scope)
    x1 = tf.contrib.layers.batch_norm(x1, 
                                      center=True, scale=True, 
                                      is_training=is_training,
                                      scope=scope+'_bn')
    x1 = tf.nn.relu(x1, name=scope+'_relu')
    return x1

# samples in [[fname, features]] format
# return X_batch, y_batch
def ym_batch_generator(samples, shapes, n_features, batch_size):
    total = len(samples)
    X_train = np.empty(shape=(total, shapes[0], shapes[1], 1))
    y_train = np.empty(shape=(total, n_features))
    for i in range(total):
        img = cv2.imread('../mouthnn_mtcnn/'+samples[i][0])
        img = cv2.resize(img, shapes, interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype('float')
        img /= 127.5
        img -= 1.
        img = img[:,:,np.newaxis]
        X_train[i] = img
        y_train[i] = samples[i][1]
        if i % 1000 == 0:
            print('%d images prepared!'%i)
    while True:
        X_batch, y_batch = shuffle(X_train, y_train)
        X_batch = X_batch[:batch_size]
        y_batch = y_batch[:batch_size]
        #X_batch = np.empty(shape=(batch_size, shapes[0], shapes[1]))
        #y_batch = np.empty(shape=(batch_size, n_features))
        #for i in range(batch_size):
        #    idx = shuffle(total)
        #    X_batch[i] = X_train[idx]
        #    y_batch[i] = y_train[idx]        
        yield X_batch, y_batch

N_FEATURES = 512
shape_used = (100, 100)
epochs = 200
batch_size = 32
n_patience = 20

f = open('../mouthnn_mtcnn/right_features.pickle', 'rb')
samples = pickle.load(f)
f.close()
#samples = samples[0:100]

train_size = len(samples)
    
train_gen = ym_batch_generator(samples, shape_used, N_FEATURES, batch_size)

tf.reset_default_graph()
is_training = tf.placeholder(dtype=tf.bool)
input_data = tf.placeholder(dtype=tf.float32, 
                           shape=[None, shape_used[0], shape_used[1], 1],
                           name='input_data')
y_true = tf.placeholder(dtype=tf.float32, 
                        shape=[None, N_FEATURES],
                        name='y_true')

x1 = conv_layer_build(input_data, 32, 'conv1')
x2 = conv_layer_build(x1, 32, 'conv2')
x3 = conv_layer_build(x2, 64, 'conv3')
x4 = conv_layer_build(x2, 64, 'conv4')
x5 = conv_layer_build(x2, 128, 'conv5')
flatten = tf.layers.flatten(x5)
y_pred = tf.layers.dense(flatten, N_FEATURES, name='output')# output layer
#loss = tf.losses.mean_squared_error(y_true, y_pred)
y_true_log = tf.log(y_true*y_true+1e-6)
y_pred_log = tf.log(y_pred*y_pred+1e-6)
loss = tf.losses.absolute_difference(y_true_log, y_pred_log)
#loss = tf.losses.mean_squared_error(y_true_log, y_pred_log)
opt = tf.train.AdamOptimizer()
update = opt.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)

train_loss = []
patience = 0
upd_per_epoch = int(train_size / batch_size)+1
best_loss = float('inf')
for i in range(epochs):
    epoch_loss = 0
    btime = time.time()
    stime = time.time()
    for j in range(upd_per_epoch):
        x_batch, y_batch = next(train_gen)       
        pred, loss_batch, _ = sess.run([y_pred, loss, update], feed_dict={
            input_data:x_batch,
            y_true:y_batch,
            is_training: True
        })        
        epoch_loss += loss_batch
        etime = time.time()
        if etime - stime > 1:
            print('\r%d/%d loss %.5f, ETA %d seconds'%(j, upd_per_epoch, epoch_loss / (j+1),
                                             int((upd_per_epoch - j) / (j+1) * (etime - btime))), end='')
            stime = time.time()
    train_loss.append(epoch_loss / upd_per_epoch)
    print('\nepoch %3d with loss %f, %d seconds'%(i, epoch_loss, int(time.time() - btime)))
    if epoch_loss < best_loss:
        print('Update best loss from '+str(best_loss)+' to %.6f.'%epoch_loss)
        best_loss = epoch_loss
        saver.save(sess, './mouthnn_ym.ckpt')
        patience = 0
    else:
        patience += 1
    if patience > n_patience:
        break

plt.plot(range(len(train_loss)), train_loss, label='training')
plt.title('Loss')
plt.legend(loc='best')
plt.savefig('train_result.png')
plt.close()
