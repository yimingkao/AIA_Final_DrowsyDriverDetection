
import pandas as pd
import numpy as np
import time
import pickle

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#set_name = 'yawn_train'
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


X_train, y_train = feature_load('dense121', 'yawn_train')
X_valid, y_valid = feature_load('dense121', 'yawn_valid')

models = [ \
          ('BaggingRegressor', BaggingRegressor(verbose=True)), \
          ('ExtraTreesRegressor', ExtraTreesRegressor(verbose=True)), \
         ]
#          ('GradientBoostingRegressor', GradientBoostingRegressor(verbose=True)), \
 #         ('RandomForestRegressor', RandomForestRegressor(verbose=True, n_jobs=3)), \
#          ('KNeighborsRegressor', KNeighborsRegressor(n_jobs=3)), \
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

npy_path = faceroi_path+'dense121_512_yawn_test/'
data = pd.read_csv(video_path+'../yawn_test.csv')
for i in range(len(data)):
    fname = data['Name'][i].replace('.avi', '.npy')
    fea = np.load(npy_path+fname)
    X = fea[:,:N_FEATURES]
    y = fea[:,N_FEATURES]
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
    plt.legend()
    plt.tight_layout()
    plt.ylabel('degree')
    plt.xlabel('Frames')
    plt.savefig(fname.replace('.npy', '.png'))
    plt.clf()
    plt.close()
    print('%d/%d: '%(i, len(data)), fname.replace('.npy', '.png') + " saved!")
    #break