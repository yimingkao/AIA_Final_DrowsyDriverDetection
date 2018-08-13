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
def isMale(name):
    return re.match('[0-9]{1,2}-Male.+', name) != None

N_FEATURES = config.N_FEATURES
extractor = config.extractor
video_path = config.video_path

setcsv_path = '../../'
window_size = 14
epoch_cnt = 20
batch_size = 32

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
        # change mark path from the original path to modified source path.
        mark_path = '../../../../YawDD/'
        if isMale(fname):
            mark_path += 'Male/'
        else:
            mark_path += 'Female/'
        mark_name = mark_path + data['Name'][i].replace('.avi', '_mark.txt')
        fmark = open(mark_name, 'r')
        degrees = np.empty(shape=fea.shape[0])
        for j in range(fea.shape[0]):
            degree = fmark.readline()
            degrees[j] = int(degree)
        fmark.close()
        
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

model = load_model(extractor+'_'+str(N_FEATURES)+'_yawn_train.h5')

for set_name in ['yawn_valid', 'yawn_test']:
    npy_path = extractor+'_'+str(N_FEATURES)+'_'+set_name+'/'
    dst_path = 'res_'+npy_path
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    
    history = []
    data = pd.read_csv(setcsv_path+set_name+'.csv')
    for i in range(len(data)):
        fname = data['Name'][i].replace('.avi', '.npy')
        fea = np.load(npy_path+fname)
        X = fea[:,:N_FEATURES]
        
        # change mark path from the original path to modified source path.
        mark_path = '../../../../YawDD/'
        if isMale(fname):
            mark_path += 'Male/'
        else:
            mark_path += 'Female/'
        mark_name = mark_path + data['Name'][i].replace('.avi', '_mark.txt')
        fmark = open(mark_name, 'r')
        y = np.empty(shape=fea.shape[0])
        for j in range(fea.shape[0]):
            degree = fmark.readline()
            y[j] = int(degree)
        fmark.close()

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