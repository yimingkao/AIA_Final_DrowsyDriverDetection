
import os
import pandas as pd
import numpy as np
import re
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import load_model

import config

N_FEATURES = config.N_FEATURES
extractor = config.extractor
video_path = config.video_path

setcsv_path = '../../'

window_size = 14

def feature_Label_load(ext, set_name):
    npy_path = ext+'_'+str(N_FEATURES)+'_'+set_name+'/'
    data = pd.read_csv(setcsv_path+set_name+'.csv')
    total = 0
    for i in range(len(data)):
        fname = data['Name'][i].replace('.avi', '.npy')
        fea = np.load(npy_path+fname)
        total += fea.shape[0]
    print(total)
    X = np.empty(shape=(total, N_FEATURES))
    y = np.empty(shape=(total))
    idx = 0
    for i in range(len(data)):
        fname = data['Name'][i].replace('.avi', '.npy')
        fea = np.load(npy_path+fname)
        X[idx:idx+fea.shape[0],:] = fea[:,:N_FEATURES]
        mark_path = '../../../../aiaDDD/markers/'
        mark_name = mark_path + data['Name'][i].replace('.avi', '.csv')
        mark = pd.read_csv(mark_name)
        degrees = mark['yawn'].values
        y[idx:idx+fea.shape[0]] = degrees[:]
        idx += fea.shape[0]
        #break;
    return X, y

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

#model = train(extractor, ['train'], 'test', model_name)
model = load_model(extractor+'_'+str(N_FEATURES)+'_train.h5')
# dump test set.
npy_path = extractor+'_'+str(N_FEATURES)+'_test/'
dst_path = 'res_' + npy_path
if not os.path.exists(dst_path):
    os.mkdir(dst_path)
    
data = pd.read_csv(setcsv_path+'test.csv')
history = []
for i in range(len(data)):
    fname = data['Name'][i].replace('.avi', '.npy')
    fea = np.load(npy_path+fname)
    X = fea[:,:N_FEATURES]

    mark_path = '../../../../aiaDDD/markers/'
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
        print(X[j:j+1].shape)
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
    
with open('lstm_'+npy_path.replace('/','.pickle.inf'), 'wb') as f:
    pickle.dump(history, f)
