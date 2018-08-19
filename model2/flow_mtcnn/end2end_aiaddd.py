
import cv2
import time
import numpy as np
import pandas as pd
import os
import re
from keras.models import load_model
import multiprocessing as mp

from mobilenet_custom_fea_extract import MobileNetCustomFeatureExtract
from mtcnn_face_det import MTCNNFaceDet

window_size = 14
N_FEATURES = 512
featureExtractor = MobileNetCustomFeatureExtract(N_FEATURES)
model = load_model('mobilecus_'+str(N_FEATURES)+'_train.h5')


video_path = '/home/jovyan/at072-group04/aiaDDD/videos/'
#marker_path = '../../YawDD/'
cord_path = '../aiaDDD/mtcnn_face/bbox/'
faceDet = MTCNNFaceDet()


check_set = 'test'
files = pd.read_csv('../aiaDDD/'+check_set+'.csv')
file_list = files['Name'].values
for fname in file_list:
    frames = 0
    fin = np.empty(shape=(window_size,N_FEATURES))
    vin = cv2.VideoCapture(video_path+fname)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vout = cv2.VideoWriter(fname.replace('.avi', '_out.avi'), fourcc, 30.0, (640,360))
    length = int(vin.get(cv2.CAP_PROP_FRAME_COUNT))
    print('{}: {}'.format(fname, length))
    cord = pd.read_csv(cord_path+fname.replace('.avi', '.csv'))
    for i in range(length):
        ret, frame = vin.read()
        if not ret:
            break
        stime = time.time()
        line = cord.iloc[i].values
        bw = line[3] - line[1]
        bh = line[4] - line[2]
        fsx = int(line[1])
        fsy = int(line[2])
        fex = int(line[3])
        fey = int(line[4])
        line = line[5:]
        if line[6]:
            sx, sy, ex, ey = faceDet.landmark2mouth(line, bw, bh)
            face_img = frame[sy:ey, sx:ex]
            face_time = time.time() - stime

            fea_time = time.time()
            features = featureExtractor.feature_extract(face_img)
            fea_time = time.time() - fea_time

            # prediction
            fin[0:window_size-1] = fin[1:window_size]
            fin[window_size-1] = features
            frames += 1
            try:
                text='face%.3f'%(1.0/face_time)
            except ZeroDivisionError:
                text = 'Inf' 
            cv2.putText(frame, text, (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            try:
                text='fea%.3f'%(1.0/fea_time)
            except ZeroDivisionError:
                text = 'Inf' 
            cv2.putText(frame, text, (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            if frames >= window_size:
                pred_time = time.time()
                pred_in = fin[np.newaxis, :]
                #pred_in = fin
                #print(pred_in.shape)
                pred = model.predict(pred_in)
                pred_time = time.time() - pred_time
                try:
                    text='pred%.3f'%(1/pred_time)
                except ZeroDivisionError:
                    text = 'Inf'
                cv2.putText(frame, text, (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.putText(frame, 'yawn: %d'%(int(pred+0.5)), (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.rectangle(frame, (fsx, fsy), (fex, fey), (0, 0, 255), 2)
                cv2.rectangle(frame, (sx, sy), (ex, ey), (255, 0, 0), 2)
        else:
            cv2.putText(frame, 'yawn: ?', (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
        vout.write(frame)
        print('\r%d'%i, end='')
    vin.release()
    vout.release()
    #break
