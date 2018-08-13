
import os
import cv2
import pandas as pd
import numpy as np
import re
import time

import config

N_FEATURES = config.N_FEATURES
extractor = config.extractor
featureExtracter = config.featureExtracter
video_path = config.video_path

#set_name = 'yawn_train'
for set_name in ['train', 'test']:
#for set_name in ['train']:
    set_path = '../../'
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

        cord = pd.read_csv('../bbox/'+data['Name'][i].replace('.avi', '_bbox.csv'))    
        fea = np.empty(shape=(length,N_FEATURES))
        for j in range(length):
            ret, frame = vin.read()
            startX = cord['sx'][j] if cord['sx'][j] >= 0 else 0
            startY = cord['sy'][j] if cord['sy'][j] >= 0 else 0
            endX = cord['ex'][j]
            endY = cord['ey'][j]         
            # Extract features by good model
            # y: 6/12 -> 11/12, use 5/12 region size
            yoff = int((endY-startY)*6/12)
            xoff = int((endX-startX)/4)
            ybot = int((endY-startY)*1/12)
            face_img = frame[startY+yoff:endY-ybot, startX+xoff:endX]
            stime = time.time()        
            pred = featureExtracter.feature_extract(face_img)
            fea[j,:] = pred
            stime = time.time()-stime
            print('\r%d %ffps'%(j, 1/stime), end='')
        vin.release()
        np.save(dst_path+data['Name'][i].replace('.avi', '.npy'), fea)
        #break