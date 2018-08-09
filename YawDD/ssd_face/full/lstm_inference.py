
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras import optimizers
from keras.models import load_model

#set_name = 'yawn_train'
video_path = '../../YawDD/Mirror/'
faceroi_path = 'ssd_face/'
N_FEATURES = 512
window_size = 7

def feature_load(ext, set_name):
    npy_path = faceroi_path+ext+'_'+str(N_FEATURES)+'_'+set_name+'/'
    data = pd.read_csv(video_path+'../'+set_name+'.csv')
    total = 0
    for i in range(len(data)):
        fname = data['Name'][i].replace('.avi', '.npy')
        fea = np.load(npy_path+fname)
        total += fea.shape[0]
        #break
    print(total)
    X = np.empty(shape=(total, N_FEATURES))
    y = np.empty(shape=(total))
    idx = 0
    for i in range(len(data)):
        fname = data['Name'][i].replace('.avi', '.npy')
        fea = np.load(npy_path+fname)
        X[idx:idx+fea.shape[0],:] = fea[:,:N_FEATURES]
        y[idx:idx+fea.shape[0]] = fea[:,N_FEATURES]
        #print(fea)
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

model = load_model('dense121_yawn_train.h5')
npy_path = faceroi_path+'dense121_512_yawn_valid/'
data = pd.read_csv(video_path+'../yawn_valid.csv')
for i in range(len(data)):
    fname = data['Name'][i].replace('.avi', '.npy')
    fea = np.load(npy_path+fname)
    X = fea[:,:N_FEATURES]
    y = fea[:,N_FEATURES]
    X, y = window_data(X, y, window_size)
    X = np.array(X)
    y = np.array(y)
    axisx = [i for i in range(len(X))]
    pred = np.array([])
    loss = 0
    for j in range(len(X)):
        o = model.predict(X[j:j+1])
        pred = np.append(pred, o)
        loss += (o - y[j])**2
    loss /= len(X)
    plt.figure(figsize=(16,7))
    plt.plot(axisx, y, label='golden')
    plt.plot(axisx, pred, label='prediction')
    plt.title('loss=%f'%loss)
    plt.legend()
    plt.tight_layout()
    plt.ylabel('degree')
    plt.xlabel('Frames')
    plt.savefig(fname.replace('.npy', '.png'))
    plt.clf()
    print('%d/%d: '%(i, len(data)), fname.replace('.npy', '.png') + " saved!")
    break