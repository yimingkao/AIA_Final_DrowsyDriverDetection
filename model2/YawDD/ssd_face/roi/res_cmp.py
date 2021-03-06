import os
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#import config

def isMale(name):
    return re.match('[0-9]{1,2}-Male.+', name) != None

extractor = 'dense121'
video_path = '../../../../../YawDD/'

fig_row = 3
fig_col = 2
fig_cnt = fig_row * fig_col


for set_name in ['yawn_valid', 'yawn_test']:
    dst_path = 'cmp_'+extractor+'_'+set_name+'/'
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    data = []
    features = [512, 1024, 2048]
    for n_features in features:
        with open('lstm_'+extractor+'_'+str(n_features)+'_'+set_name+'.pickle', 'rb') as f:
            data.append(pickle.load(f))
    
    pic = 0
    history = []
    for i in range(len(data[0])):
        fname = data[0][i][2]
        mark_path = '../../../../YawDD/'
        if isMale(fname):
            mark_path += 'Male/'
        else:
            mark_path += 'Female/'
        mark_name = mark_path + fname.replace('.avi', '_mark.txt')
        fmark = open(mark_name, 'r')
        y = np.empty(shape=data[0][i][0].shape[0])
        for j in range(len(y)):
            degree = fmark.readline()
            y[j] = int(degree)
        fmark.close()
        nth_pic = pic % fig_cnt
        if nth_pic == 0:
            plt.figure(figsize=(16,10))
        plt.subplot(fig_row,fig_col,nth_pic+1)
        axisx = [i for i in range(len(y))]
        plt.plot(axisx, y, label='golden')
        entry = [fname]
        for j in range(len(features)):
            label = str(features[j])+','+'%.2f'%data[j][i][1][0]
            plt.plot(axisx, data[j][i][0], label=label)
            entry.extend(data[j][i][1][0])
        history.append(entry)
        plt.title(fname)
        plt.legend()
        plt.tight_layout()
        plt.ylabel('degree')
        plt.xlabel('Frames')
        print('%d/%d: '%(i, len(data[0])), fname.replace('.avi', '.png') + " saved!")
        if nth_pic == fig_cnt-1:
            plt.savefig(dst_path+'plots_'+str(int(pic/fig_cnt))+'.png')
            plt.close()
        pic += 1
    if nth_pic != fig_cnt-1:
        plt.savefig(dst_path+'plots_'+str(int(pic/fig_cnt))+'.png')
        plt.close()

    #print(history)
    col_name = ['name']
    for j in range(len(features)):
        col_name.append(str(features[j]))
    loss = pd.DataFrame(history, columns=col_name)
    loss.to_csv(set_name+'_cmp.csv', index=False)