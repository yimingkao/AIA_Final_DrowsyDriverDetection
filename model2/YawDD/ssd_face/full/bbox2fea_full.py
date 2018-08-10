
import os
import cv2
import pandas as pd
import numpy as np
import re
import time

from dense121_fea_extract import Dense121FeatureExtract

def isMale(name):
    return re.match('[0-9]{1,2}-Male.+', name) != None

N_FEATURES = 512
extractor = 'dense121'
featureExtracter = Dense121FeatureExtract(N_FEATURES)
#video_path = '/projectdata/driver/YawDD/'
video_path = '../../../../../YawDD/'

#set_name = 'yawn_train'
for set_name in ['yawn_train', 'yawn_valid', 'yawn_test']:
    set_path = '../bbox/'+set_name+'/'
    dst_path = extractor + '_' + str(N_FEATURES) + '_' + set_name + '/'
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)

    data = pd.read_csv('../../'+set_name+'.csv')
    for i in range(len(data)):
        target_path = video_path
        if isMale(data['Name'][i]):
            target_path += 'Male/'
        else:
            target_path += 'Female/'
        filename = target_path + data['Name'][i]
        txtname = filename.replace('.avi', '_mark.txt')
        vin = cv2.VideoCapture(filename)
        length = int(vin.get(cv2.CAP_PROP_FRAME_COUNT))
        print('{}: {}'.format(filename, length))
#        fmark = open(txtname, 'r')
#        degrees = []
#        for j in range(length):
#            degree = fmark.readline()
#            degrees.append(int(degree))
#        fmark.close()
        
        cord = pd.read_csv(set_path+data['Name'][i].replace('.avi', '.csv'))    
        fea = np.empty(shape=(length,N_FEATURES))
        for j in range(length):
            ret, frame = vin.read()
            startX = cord['sx'][j] if cord['sx'][j] >= 0 else 0
            startY = cord['sy'][j] if cord['sy'][j] >= 0 else 0
            endX = cord['ex'][j]
            endY = cord['ey'][j]         
            # Extract features by good model
            face_img = frame[startY:endY, startX:endX]
            #cv2.imwrite('test.jpg', face_img)
            stime = time.time()        
            pred = featureExtracter.feature_extract(face_img)
            fea[j,:] = pred
            stime = time.time()-stime
            print('\r%d %ffps'%(j, 1/stime), end='')
            #if j == 10:
            #    break
        #print(fea)
        vin.release()
        np.save(dst_path+data['Name'][i].replace('.avi', '.npy'), fea)
        #break