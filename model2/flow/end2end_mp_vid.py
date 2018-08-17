
import cv2
import time
import numpy as np
import pandas as pd
import os
from keras.models import load_model
import multiprocessing as mp

from ssd_face_det import SSDFaceDet
#from dense121_fea_extract import Dense121FeatureExtract
#from mobilenet_fea_extract import MobileNetFeatureExtract
from mobilenet_custom_fea_extract import MobileNetCustomFeatureExtract

video_path = '/home/jovyan/at072-group04/aiaDDD/videos/'
marker_path = '/home/jovyan/at072-group04/aiaDDD/markers/'
cord_path = '../aiaDDD/ssd_face/bbox/'

def faceFinder(vstop, predQ, faceQ):
    faceDet = SSDFaceDet()
    faceDet.reset()
    while vstop.value == 0:
        try:
            frame = faceQ.get(1000)
            startX, startY, endX, endY = faceDet.faceFind(frame)
            predQ.put((startX, startY, endX, endY))
        except:
            pass

def predictor(vstop, predQ, faceQ):
    window_size = 14
    N_FEATURES = 512
    #featureExtractor = Dense121FeatureExtract(N_FEATURES)
    #featureExtractor = MobileNetFeatureExtract(N_FEATURES)
    featureExtractor = MobileNetCustomFeatureExtract(N_FEATURES)
    model = load_model('mobilecus_'+str(N_FEATURES)+'_train.h5')
    tracker = cv2.TrackerMedianFlow_create()
    
    targets = []
    for root,subdir,files in os.walk(video_path):
        targets = files
    for fname in targets:
        frames = 0
        fin = np.empty(shape=(window_size,N_FEATURES))
        vin = cv2.VideoCapture(video_path+fname)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        vout = cv2.VideoWriter(fname.replace('.avi', '_out.avi'), fourcc, 30.0, (640,360))
        length = int(vin.get(cv2.CAP_PROP_FRAME_COUNT))
        print('{}: {}'.format(fname, length))
        cord = pd.read_csv(cord_path+fname.replace('.avi', '_bbox.csv'))
        marker = pd.read_csv(marker_path+fname.replace('.avi', '.csv'))
        degree = marker['yawn'].values
        for i in range(length):
            ret, frame = vin.read()
            if not ret:
                break
            stime = time.time()
#            if frames == 0:
#                faceQ.put(frame)           
#                (startX, startY, endX, endY) = predQ.get()
#                tracker.clear()
#                tracker = cv2.TrackerMedianFlow_create()
#                tracker.init(frame, (startX, startY, endX-startX, endY-startY))
#            else:
#                if frames == 1:
#                    faceQ.put(frame)
#                try:
#                    (startX, startY, endX, endY) = predQ.get(False)
#                    faceQ.put(frame)
#                    tracker.clear()
#                    tracker = cv2.TrackerMedianFlow_create()
#                    tracker.init(frame, (startX, startY, endX-startX, endY-startY))
#                except:
#                    (success, box) = tracker.update(frame)
#                    if success:
#                        (startX, startY, w, h) = [int(v) for v in box]
#                        endX = startX + w
#                        endY = startY + h
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
                if degree[i] != int(pred+0.5):
                    cv2.putText(frame, 'yawn: %d'%(int(pred+0.5)), (20, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.putText(frame, 'yawn: %d'%(degree[i]), (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.rectangle(frame, (startX+xoff, startY+yoff), (endX, endY-ybot), (255, 0, 0), 2)
                cv2.rectangle(frame, (cord['sx'][i], cord['sy'][i]),
                              (cord['ex'][i], cord['ey'][i]), (0, 255, 0), 2)
            vout.write(frame)
            print('\r%d'%i, end='')
        vin.release()
        vout.release()
        #break #for fname in targets:
    faceQ.put(frame)
    vstop.value = 1
    cv2.destroyAllWindows()

if __name__ == '__main__':
    vstop = mp.Value('i', 0)  #
    predQ = mp.Queue()
    faceQ = mp.Queue()
    p1 = mp.Process(target=predictor, args=(vstop, predQ, faceQ)) #偵測疲勞
    p2 = mp.Process(target=faceFinder, args=(vstop, predQ, faceQ)) #找臉
    p1.start()
    p2.start()
    p1.join()
    p2.join()
