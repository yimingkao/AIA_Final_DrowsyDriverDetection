import os
import cv2
import gc
import numpy as np
import random
from sklearn.utils import shuffle 
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time


def img_op_shift(x, ratio):
    nx = np.empty(shape=x.shape)
    lx = x.shape[1]
    ly = x.shape[0]
    ofsx = lx * random.randint(0, 20)/100
    ofsy = ly * random.randint(0, 20)/100
    dx = lx - ofsx
    dy = ly - ofsy
    if random.randint(0, 20) % 2 == 0:
        nx[0:dy,0:dx] = x[ofsy:, ofsx:]
    else:
        nx[ofsy:,ofsx:] = x[0:dy, 0:dx]
    return nx

def img_op_scale(x, shape):
    lx = x.shape[1]
    ly = x.shape[0]
    ox = lx * random.randint(0, 10)/100
    oy = ly * random.randint(0, 10)/100
    x = x[oy:ly-2*oy,ox:lx-2*ox]
    x = cv2.resize(x, shape)
    return x

def conv_layer_build(x_tensor, num, scope):
    x1 = tf.layers.conv2d(x_tensor, num, kernel_size=(3,3), strides=(2,2),
                          name=scope)
    x1 = tf.contrib.layers.batch_norm(x1, 
                                      center=True, scale=True, 
                                      is_training=is_training,
                                      scope=scope+'_bn')
    x1 = tf.nn.relu(x1, name=scope+'_relu')
    return x1

# return X_batch, y_batch
def ym_batch_generator(path, shapes, n_features, batch_size):
    total = 0
    for i in range(6):
        src = path+str(i)
        for root,subdir,files in os.walk(src):
            total += len(files)
    print(total)
    while True:
        X_train = 0
        y_train = 0
        gc.collect()
        X_train = np.empty(shape=(total, shapes[0], shapes[1], 1))
        y_train = np.empty(shape=(total,1))
        idx = 0
        for i in range(6):
            src = path+str(i)
            for root,subdir,files in os.walk(src):
                for j, file in enumerate(files):
                    img = cv2.imread(root+'/'+file)
                    img = cv2.resize(img, shapes, interpolation=cv2.INTER_AREA)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = img.astype('float')
                    img /= 127.5
                    img -= 1.
                    img = img[:,:,np.newaxis]
                    X_train[idx+j] = img
                    y_train[idx+j] = i
                idx += len(files)
        emit = 0
        X_train, y_train = shuffle(X_train, y_train)
        while emit < total:
            end = emit + batch_size
            if end <= total:
                X_batch = X_train[emit:end]
                y_batch = y_train[emit:end]
            else:
                X_batch = X_train[total-batch_size:]
                y_batch = y_train[total-batch_size:]
            emit += batch_size
            yield X_batch, y_batch
            
def testset_load(path, shapes):
    total = 0
    for i in range(6):
        src = path+str(i)
        for root,subdir,files in os.walk(src):
            total += len(files)
    print(total)
    X_test = np.empty(shape=(total, shapes[0], shapes[1], 1))
    y_test = np.empty(shape=(total,1))
    idx = 0
    for i in range(6):
        src = path+str(i)
        for root,subdir,files in os.walk(src):
            for j, file in enumerate(files):
                img = cv2.imread(root+'/'+file)
                img = cv2.resize(img, shapes, interpolation=cv2.INTER_AREA)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = img.astype('float')
                img /= 127.5
                img -= 1.
                img = img[:,:,np.newaxis]
                X_test[idx+j] = img
                y_test[idx+j] = i
            idx += len(files)
    return X_test, y_test

N_FEATURES = 512
shape_used = (100, 100)
epochs = 200
batch_size = 32
n_patience = 20

train_size = 10657
train_gen = ym_batch_generator('../mouthnn_mtcnn/train/', shape_used, N_FEATURES, batch_size)
#X_test, y_test = testset_load('../mouthnn_mtcnn/train/', shape_used)

tf.reset_default_graph()
is_training = tf.placeholder(dtype=tf.bool)
input_data = tf.placeholder(dtype=tf.float32, 
                           shape=[None, shape_used[0], shape_used[1], 1],
                           name='input_data')
y_true = tf.placeholder(dtype=tf.float32, 
                        shape=[None, 1],
                        name='y_true')

x1 = conv_layer_build(input_data, 32, 'conv1')
x2 = conv_layer_build(x1, 32, 'conv2')
x3 = conv_layer_build(x2, 64, 'conv3')
x4 = conv_layer_build(x3, 64, 'conv4')
x5 = conv_layer_build(x4, 128, 'conv5')
flatten = tf.layers.flatten(x5)
fea_out = tf.layers.dense(flatten, N_FEATURES, name='fea_out')
#flatten2 = tf.layers.flatten(fea_out)
y_pred = tf.layers.dense(fea_out, 1, name='output') # output layer
loss = tf.losses.mean_squared_error(y_true, y_pred)
#loss = tf.losses.absolute_difference(y_true_log, y_pred_log)
#loss = tf.losses.mean_squared_error(y_true_log, y_pred_log)
opt = tf.train.AdamOptimizer(learning_rate=1e-5)
update = opt.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)

train_loss = []
valid_loss = []
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
            print('\r%3d: %d/%d loss %.5f, ETA %d seconds'%(i, j, upd_per_epoch, epoch_loss / (j+1),
                                             int((upd_per_epoch - j) / (j+1) * (etime - btime))), end='')
            stime = time.time()
#    pred, test_loss = sess.run([y_pred, loss], feed_dict={
#        input_data: X_test,
#        y_true: y_test,
#        is_training: False
#    })
#    print(pred.shape)
    train_loss.append(epoch_loss / upd_per_epoch)
#    valid_loss.append(test_loss)
#    print('\nepoch %3d with train loss %f, valid loss %f, %d sec'%(i, epoch_loss, test_loss, int(time.time() - btime)))
    print('\nepoch %3d with loss %f, %d sec'%(i, epoch_loss, int(time.time() - btime)))
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
#plt.plot(range(len(train_loss)), valid_loss, label='test')
plt.title('Loss')
plt.legend(loc='best')
plt.savefig('train_result.png')
plt.close()
