
import pandas as pd
import numpy as np
import time
import pickle

from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

video_path = '../../YawDD/Mirror/'
faceroi_path = 'ssd_face/'
N_FEATURES = 512

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


X_train_raw, y_train_raw = feature_load('dense121', 'yawn_train')
X_valid, y_valid = feature_load('dense121', 'yawn_valid')
counts = [0, 0, 0, 0, 0, 0]
for i in range(len(y_train_raw)):
    counts[int(y_train_raw[i])] += 1
print(counts)
print(np.min(np.array(counts)))
each_size = np.min(np.array(counts))

X_groups = [np.empty(shape=(counts[i], N_FEATURES)) for i in range(len(counts))]
y_groups = [np.empty(shape=(counts[i])) for i in range(len(counts))]
idxes = [0, 0, 0, 0, 0, 0]
for i in range(len(y_train_raw)):
    group = int(y_train_raw[i])
    idx = idxes[group]
    X_groups[group][idx:idx+1,:] = X_train_raw[i,:]
    y_groups[group][idx:idx+1] = y_train_raw[i]
    idxes[group] += 1

for i in range(len(X_groups)):
    X_groups[i], y_groups[i] = shuffle(X_groups[i], y_groups[i], random_state=i)

X_train = np.empty(shape=(each_size*6, N_FEATURES))
y_train = np.empty(shape=(each_size*6))
idx = 0
for i in range(len(X_groups)):
    X_train[idx:idx+each_size,:] = X_groups[i][0:each_size,:]
    y_train[idx:idx+each_size] = y_groups[i][0:each_size]
    idx += each_size
    
xscaler = StandardScaler().fit(X_train)
X_train = xscaler.transform(X_train)
y_train = y_train/5

models = [ \
          ('BaggingRegressor', BaggingRegressor()), \
          ('ExtraTreesRegressor', ExtraTreesRegressor()), \
         ]
#          ('GradientBoostingRegressor', GradientBoostingRegressor(verbose=True)), \
#          ('RandomForestRegressor', RandomForestRegressor(verbose=True)), \
#          ('KNeighborsRegressor', KNeighborsRegressor()), \
#          ('DecisionTreeRegressor', DecisionTreeRegressor()), \
#          ('AdaBoostRegressor', AdaBoostRegressor()), \


min_score = 1e20
power_res = []
for name, model in models:
    stime = time.time()
    model.fit(X_train, y_train)
    ypred = model.predict(X_valid)
    score = np.sqrt(((ypred - y_valid)**2).sum())
    stime = time.time() - stime
    print(name + " finished!")
    if score < min_score:
        min_score = score
        power_best_model = model
    power_res += [[name, score, stime]]

power_df = pd.DataFrame(power_res, columns=['name', 'score', 'time'])
print(power_df)
with open('power_best_model.pickle', 'wb') as f:
    pickle.dump(power_best_model, f)

npy_path = faceroi_path+'dense121_512_yawn_train/'
data = pd.read_csv(video_path+'../yawn_train.csv')
for i in range(len(data)):
    fname = data['Name'][i].replace('.avi', '.npy')
    fea = np.load(npy_path+fname)
    X = fea[:,:N_FEATURES]
    y = fea[:,N_FEATURES]
    X = xscaler.transform(X)
    axisx = [i for i in range(len(X))]
    pred = np.array([])
    loss = 0
    for j in range(len(X)):
        o = model.predict(X[j:j+1])
        o = o * 5
        pred = np.append(pred, o)
        loss += (o - y[j])**2
    loss /= len(X)
    plt.figure(figsize=(16,7))
    plt.plot(axisx, y, label='golden')
    plt.plot(axisx, pred, label='prediction')
    plt.legend()
    plt.tight_layout()
    plt.ylabel('degree')
    plt.xlabel('Frames')
    plt.savefig(fname.replace('.npy', '.png'))
    plt.clf()
    plt.close()
    print('%d/%d: '%(i, len(data)), fname.replace('.npy', '.png') + " saved!")
    #break