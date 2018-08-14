import os
import pickle
import pandas as pd
import numpy as np
import cv2


extractor = 'mobilenet'
window_size = 14

features = [512, 1024, 2048]

for n_features in features:
    dst_path = 'vid_'+extractor+'_'+str(n_features)+'_test/'
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
        
    with open('lstm_'+extractor+'_'+str(n_features)+'_test.pickle', 'rb') as f:
        data = pickle.load(f)
    for i in range(len(data)):
        fname = data[i][2]
        vid_path = '../../../../../../at072-group04/aiaDDD/'
        vin = cv2.VideoCapture(vid_path+fname)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        vout = cv2.VideoWriter(dst_path+fname, fourcc, 30.0, (640,360))
        length = int(vin.get(cv2.CAP_PROP_FRAME_COUNT))
        print(fname + ": %d"%length)

        mark_path = '../../../../aiaDDD/markers/'
        mark_name = mark_path + fname.replace('.avi', '.csv')
        mark = pd.read_csv(mark_name)
        y = mark['yawn'].values
        pred = np.empty(shape=(length))
        pred[window_size:] = data[i][0]
        
        for j in range(length):
            ret, frame = vin.read()
            if y[j] != int(pred[j]+0.5):
                cv2.putText(frame, 'yawn: %d'%(int(pred[j]+0.5)), (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)   
            cv2.putText(frame, 'yawn: %d'%y[j], (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            vout.write(frame) 
        
    vin.release()
    vout.release()
