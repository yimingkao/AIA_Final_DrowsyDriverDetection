
import os
import cv2
import pandas as pd
import numpy as np
import time
from mtcnn_face_det import MTCNNFaceDet

import config

N_FEATURES = config.N_FEATURES
extractor = config.extractor
featureExtractor = config.featureExtractor
video_path = config.video_path

faceDet = MTCNNFaceDet()

for set_name in ['train', 'test']:
#for set_name in ['train']:
    set_path = '../'
    dst_path = extractor + '_' + str(N_FEATURES) + '_' + set_name + '/'
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    
    data = pd.read_csv(set_path+set_name+'.csv')
    for i in range(len(data)):
        target_path = video_path
        filename = target_path + data['Name'][i]
        vin = cv2.VideoCapture(filename)
        length = int(vin.get(cv2.CAP_PROP_FRAME_COUNT))
        print('{}: {}'.format(filename, length))

        cord = pd.read_csv('bbox/'+data['Name'][i].replace('.avi', '.csv'))    
        fea = np.empty(shape=(length,N_FEATURES))
        index = 0
        for j in range(length):
            ret, frame = vin.read()
            line = cord.iloc[j].values
            bw = line[3] - line[1]
            bh = line[4] - line[2]
            line = line[5:]
            if line[6]:
                sx, sy, ex, ey = faceDet.landmark2mouth(line, bw, bh)
            else:
                continue
            face_img = frame[sy:ey, sx:ex]
            stime = time.time()        
            pred = featureExtractor.feature_extract(face_img)
            fea[index,:] = pred
            index += 1
            stime = time.time()-stime
            print('\r%d %ffps'%(j, 1/stime), end='')
        fea = fea[:index]
        vin.release()
        np.save(dst_path+data['Name'][i].replace('.avi', '.npy'), fea)
        #break
    #break
