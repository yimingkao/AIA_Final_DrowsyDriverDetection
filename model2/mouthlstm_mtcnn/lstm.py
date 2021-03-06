import os
import pandas as pd
import numpy as np
import re
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.models import Sequential
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

def feature_Label_load(ext, set_name, mark_path, bbox_path):
    npy_path = ext+'_'+str(N_FEATURES)+'_'+set_name+'/'
    data = pd.read_csv(set_name+'.csv')
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
        mark_name = mark_path + data['Name'][i].replace('.avi', '.csv')
        mark = pd.read_csv(mark_name)
        # The bbox might has empty record at beginning lines so we have to check!
        bbox_name = bbox_path + data['Name'][i].replace('.avi', '.csv')
        bbox = pd.read_csv(bbox_name)
        lx3 = bbox['lx3'].values
        for j in range(len(lx3)):
            if lx3[j] != 0:
                break
        degrees = mark['yawn'].values
        y[idx:idx+fea.shape[0]] = degrees[j:]
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

# param[in] ext (string) The feature extractor name
# param[in] trains (list) The list of strings contains the sets used as training data.
# param[in] tpaths (list) The list of strings contains the path of sets
# param[in] bpaths (list) The list of strings contains the path of bounding boxes.
# param[in] valid (list) The validation set name, path, bounding boxes
# param[in] save (string) The save filename includes path of model
def train(ext, trains, tpaths, bpaths, valid, save):
    X_train_raw, y_train_raw = feature_Label_load(ext, trains[0], tpaths[0], bpaths[0])
    for idx, train in enumerate(trains[1:]):
        X_load, y_load = feature_Label_load(ext, train, tpaths[idx+1], bpaths[idx+1])
        X_train_raw = np.concatenate((X_train_raw, X_load), axis=0)
        y_train_raw = np.concatenate((y_train_raw, y_load))
    X_train, y_train = window_data(X_train_raw, y_train_raw, window_size)
    X_valid_raw, y_valid_raw = feature_Label_load(ext, valid[0], valid[1], valid[2])
    X_valid, y_valid = window_data(X_valid_raw, y_valid_raw, window_size)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_valid = np.array(X_valid)
    y_valid = np.array(y_valid)
    #print(X_train, y_train)
    
    model = Sequential()
    model.add(NN_MODEL(512, input_shape=(window_size, N_FEATURES)))
    model.add(Dense(1))
    print(model.summary())
    opt = optimizers.Adam(clipvalue=5)   
    model.compile(loss='mean_squared_error',
                 optimizer=opt,
                 metrics=['mae'])
    checkpoint = ModelCheckpoint(save, monitor='val_loss', verbose=1,
                                 save_best_only=True)
    earlystop = EarlyStopping(monitor='val_loss', patience=n_patience, verbose=1)
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epoch_cnt,
              validation_data=(X_valid, y_valid),
              callbacks=[checkpoint, earlystop])
#              callbacks=[earlystop])
    return model

model_name = extractor+'_'+str(N_FEATURES)+'_train.h5'
model = train(extractor,
              ['train', 'yawn_train'], 
              ['../../aiaDDD/markers/', '../../YawDD/markers/'],
              ['../aiaDDD/mtcnn_face/bbox/', '../YawDD/mtcnn_face/bbox/'],
              ['test', '../../aiaDDD/markers/', '../aiaDDD/mtcnn_face/bbox/'],
              model_name)

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
