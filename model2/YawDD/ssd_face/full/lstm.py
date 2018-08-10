
import pandas as pd
import numpy as np
import re
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras import optimizers

def isMale(name):
    return re.match('[0-9]{1,2}-Male.+', name) != None

#video_path = '/projectdata/driver/YawDD/'
video_path = '../../../../../YawDD/'
setcsv_path = '../../'
N_FEATURES = 2048
extractor = 'dense121'
window_size = 14

def feature_load(ext, set_name):
    npy_path = ext+'_'+str(N_FEATURES)+'_'+set_name+'/'
    data = pd.read_csv(setcsv_path+set_name+'.csv')
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
        mark_path = video_path
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

def train(ext, train, valid):
    X_train_raw, y_train_raw = feature_load(ext, train)
    X_train, y_train = window_data(X_train_raw, y_train_raw, window_size)
    X_valid_raw, y_valid_raw = feature_load(ext, valid)
    X_valid, y_valid = window_data(X_valid_raw, y_valid_raw, window_size)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_valid = np.array(X_valid)
    y_valid = np.array(y_valid)
    #print(X_train, y_train)
    
    model = Sequential()
    model.add(LSTM(512, input_shape=(window_size, N_FEATURES)))
    model.add(Dense(1))
    print(model.summary())
    opt = optimizers.Adam(clipvalue=5)
    model.compile(loss='mean_squared_error',
                 optimizer=opt,
                 metrics=['mae'])
    model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_valid, y_valid))
    return model

model = train(extractor, 'yawn_train', 'yawn_train')
model.save(extractor+'_'+str(N_FEATURES)+'_yawn_train.h5')
npy_path = extractor+'_'+str(N_FEATURES)+'_yawn_train/'
data = pd.read_csv(setcsv_path+'yawn_train.csv')
history = []
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
    history.append([pred, loss, data['Name'][i]])
    plt.figure(figsize=(16,7))
    plt.plot(axisx, y, label='golden')
    plt.plot(axisx, pred, label='prediction')
    plt.title('loss: %f'%loss)
    plt.legend()
    plt.tight_layout()
    plt.ylabel('degree')
    plt.xlabel('Frames')
    plt.savefig(fname.replace('.npy', '.png'))
    plt.clf()
    plt.close()
    print('%d/%d: '%(i, len(data)), fname.replace('.npy', '.png') + " saved!")
    #break
    
with open('lstm_'+npy_path.replace('/','.pickle'), 'wb') as f:
    pickle.dump(history, f)