import os
import pickle
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#import config

extractors = ['dense121', 'mobilenet']
features = [512, 1024, 2048]
#features = [2048]
window_size = 14
fig_row = 3
fig_col = 2
fig_cnt = fig_row * fig_col


for set_name in ['test']:
    dst_path = 'cmp_'+set_name+'/'
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    data = []
    for extractor in extractors:
        for n_features in features:
            with open('lstm_'+extractor+'_'+str(n_features)+'_'+set_name+'.pickle', 'rb') as f:
                data.append(pickle.load(f))
    pic = 0
    history = []
    for i in range(len(data[0])):
        fname = data[0][i][2]
        mark_path = '../../../../aiaDDD/markers/'
        mark_name = mark_path + fname.replace('.avi', '.csv')
        mark = pd.read_csv(mark_name)
        y = mark['yawn'].values
        y = y[window_size:]
        nth_pic = pic % fig_cnt
        if nth_pic == 0:
            plt.figure(figsize=(16,10))
        plt.subplot(fig_row,fig_col,nth_pic+1)
        axisx = [i for i in range(len(y))]
        plt.plot(axisx, y, label='golden')
        entry = [fname]
        idx = 0
        for extractor in extractors:
            for j in range(len(features)):
                label = extractor+str(features[j])+','+'%.2f'%data[idx][i][1][0]
                plt.plot(axisx, data[idx][i][0], label=label)
                entry.extend(data[idx][i][1][0])
                idx += 1
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
    for extractor in extractors:
        for j in range(len(features)):
            col_name.append(extractor + str(features[j]))
    loss = pd.DataFrame(history, columns=col_name)
    loss.to_csv(set_name+'_cmp.csv', index=False)
