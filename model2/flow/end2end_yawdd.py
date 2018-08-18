
import cv2
import time
import numpy as np
import pandas as pd
import os
import re
from keras.models import load_model
import multiprocessing as mp

from mobilenet_custom_fea_extract import MobileNetCustomFeatureExtract

def isMale(name):
    return re.match('[0-9]{1,2}-Male.+', name) != None

window_size = 14
N_FEATURES = 512
featureExtractor = MobileNetCustomFeatureExtract(N_FEATURES)
model = load_model('mobilecus_'+str(N_FEATURES)+'_train.h5')


video_path = '/projectdata/driver/YawDD/'
#marker_path = '../../YawDD/'
cord_path = '../YawDD/ssd_face/bbox/'

check_set = 'yawn_valid'
files = pd.read_csv('../YawDD/'+check_set+'.csv')
file_list = files['Name'].values
cord_path = cord_path + check_set + '/'
for fname in file_list:
    frames = 0
    fin = np.empty(shape=(window_size,N_FEATURES))
    src_path = video_path
    if isMale(fname):
        src_path += 'Male/'
    else:
        src_path += 'Female/'
    vin = cv2.VideoCapture(src_path+fname)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vout = cv2.VideoWriter(fname.replace('.avi', '_out.avi'), fourcc, 30.0, (640,480))
    length = int(vin.get(cv2.CAP_PROP_FRAME_COUNT))
    print('{}: {}'.format(fname, length))
    cord = pd.read_csv(cord_path+fname.replace('.avi', '.csv'))
    for i in range(length):
        ret, frame = vin.read()
        if not ret:
            break
        stime = time.time()
        startX = cord['sx'][i]
        startY = cord['sy'][i]
        endX   = cord['ex'][i]
        endY   = cord['ey'][i]
        face_time = time.time() - stime
        #cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

        # Extract features by good model
        startX = int(startX)
        startY = int(startY)
        endX = int(endX)
        endY = int(endY)
        # y: 6/12 -> 11/12, use 5/12 region size
        yoff = int((endY-startY)*6/12)
        xoff = int((endX-startX)/4)
        ybot = int((endY-startY)*1/12)
        face_img = frame[startY+yoff:endY-ybot, startX+xoff:endX]
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
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.rectangle(frame, (startX+xoff, startY+yoff), (endX, endY-ybot), (255, 0, 0), 2)
        vout.write(frame)
        print('\r%d'%i, end='')
    vin.release()
    vout.release()
    #break
