
import pandas as pd
import numpy as np
import re
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.models import load_model
def isMale(name):
    return re.match('[0-9]{1,2}-Male.+', name) != None

#video_path = '/projectdata/driver/YawDD/'
video_path = '../../../../../YawDD/'
setcsv_path = '../../'
N_FEATURES = 2048
extractor = 'dense121'
window_size = 14

def feature_load(ext, fname, npy_path):
    # X
    fea = np.load(npy_path+fname)
    X = np.empty(shape=(fea.shape[0], N_FEATURES))
    X = fea[:,:N_FEATURES]
    
    # y
    y = np.empty(shape=(fea.shape[0]))
    mark_path = video_path
    if isMale(fname):
        mark_path += 'Male/'
    else:
        mark_path += 'Female/'
    mark_name = mark_path + data['Name'][i].replace('.avi', '_mark.txt')
    fmark = open(mark_name, 'r')
    for j in range(fea.shape[0]):
        degree = fmark.readline()
        y[j] = int(degree)
    fmark.close()
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
npy_path = extractor+'_'+str(N_FEATURES)+'_yawn_train/'
data = pd.read_csv(setcsv_path+'yawn_train.csv')
history = []
for i in range(len(data)):
    fname = data['Name'][i].replace('.avi', '.npy')
    X, y = feature_load(extractor, fname, npy_path)
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