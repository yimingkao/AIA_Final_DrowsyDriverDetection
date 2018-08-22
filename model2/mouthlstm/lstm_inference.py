import os
import pandas as pd
import numpy as np
import re
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, GRU
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

N_FEATURES = 512
extractor = 'mobilecus'

window_size = 14
epoch_cnt = 150
n_patience = 15
batch_size = 32
NN_MODEL = LSTM #GRU

def window_data(data, label, window_size):
    X = []
    y = []
    i = 0
    while (i + window_size) <= len(data) - 1:
        X.append(data[i:i+window_size])
        y.append([label[i+window_size]])
        i += 1
    assert len(X) == len(y)
    return X, y

model_name = extractor+'_'+str(N_FEATURES)+'_train.h5'
model = load_model(model_name)

# dump test set.
npy_path = extractor+'_'+str(N_FEATURES)+'_test/'
dst_path = 'res_' + npy_path
if not os.path.exists(dst_path):
    os.mkdir(dst_path)
    
data = pd.read_csv('test.csv')
history = []
for i in range(len(data)):
    fname = data['Name'][i].replace('.avi', '.npy')
    fea = np.load(npy_path+fname)
    X = fea[:,:N_FEATURES]

    mark_path = '../../aiaDDD/markers/'
    mark_name = mark_path + data['Name'][i].replace('.avi', '.csv')
    mark = pd.read_csv(mark_name)
    y = mark['yawn'].values

    X, y = window_data(X, y, window_size)
    X = np.array(X)
    y = np.array(y)
    axisx = [i for i in range(len(X))]
    pred = np.array([])
    loss = 0
    print(X.shape)
    print(X[0].shape)
    for j in range(len(X)):
        o = model.predict(X[j:j+1])
        pred = np.append(pred, o)
        loss += (o - y[j])**2
    loss /= len(X)
    history.append([pred, loss, data['Name'][i]])
    plt.figure(figsize=(16,7))
    plt.plot(axisx, y, label='golden')
    plt.plot(axisx, pred, label='prediction')
    plt.title('loss: %f'%loss)
    plt.legend()
    plt.tight_layout()
    plt.ylabel('degree')
    plt.xlabel('Frames')
    plt.savefig(dst_path+fname.replace('.npy', '.png'))
    plt.clf()
    plt.close()
    print('%d/%d: '%(i, len(data)), fname.replace('.npy', '.png') + " saved!")
    
with open('lstm_'+npy_path.replace('/','.pickle'), 'wb') as f:
    pickle.dump(history, f)
