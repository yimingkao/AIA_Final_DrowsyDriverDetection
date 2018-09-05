
import cv2
import time
import numpy as np
import pandas as pd
from keras.models import load_model
import multiprocessing as mp

from mtcnn_face_det import MTCNNFaceDet
from mobilenet_custom_fea_extract import MobileNetCustomFeatureExtract
TIME_PERIOD = 100 # frames to check the detection/feature extract/pred time.

def faceFinder(vstop, predQ, faceQ):
    faceDet = MTCNNFaceDet()
    faceDet.reset()
    face_time = []
    while vstop.value == 0:
        try:
            frame = faceQ.get(True, 1)
            stime = time.time()
            startX, startY, endX, endY, landmarks = faceDet.faceFind(frame)
            face_time.append(time.time() - stime)
            ftime = 0
            for i in range(len(face_time)):
                ftime += face_time[i]
            ftime = ftime / len(face_time)
            predQ.put_nowait((startX, startY, endX, endY, landmarks, ftime))
            if len(face_time) >= TIME_PERIOD:
                face_time.pop()
        except:
            pass

def predictor(vstop, predQ, faceQ):
    window_size = 14
    N_FEATURES = 512
    faceDet = MTCNNFaceDet()
    featureExtractor = MobileNetCustomFeatureExtract(N_FEATURES)
    model = load_model('mobilecus_'+str(N_FEATURES)+'_train.h5')
    tracker = cv2.TrackerMedianFlow_create()
    
    frames = 0
    fin = np.empty(shape=(window_size,N_FEATURES))
    vin = cv2.VideoCapture(0)
    fea_logs = []  # feature extraction time logs
    pred_logs = [] # predition time logs
    while True:
        ret, frame = vin.read()
        if not ret:
            break
        if frames == 0:
            faceQ.put(frame)           
            (startX, startY, endX, endY, landmarks, face_time) = predQ.get()
            if endX == 0 or endY == 0:
                continue
            tracker.clear()
            tracker = cv2.TrackerMedianFlow_create()
            tracker.init(frame, (startX, startY, endX-startX, endY-startY))
        else:
            if frames == 1:
                faceQ.put(frame)
            try:
                (startX, startY, endX, endY, landmarks, face_time) = predQ.get(False)
                faceQ.put(frame)
                tracker.clear()
                tracker = cv2.TrackerMedianFlow_create()
                tracker.init(frame, (startX, startY, endX-startX, endY-startY))
            except:
                (success, box) = tracker.update(frame)
                if success:
                    (startX, startY, w, h) = [int(v) for v in box]
                    endX = startX + w
                    endY = startY + h
    
        # Extract features by good model
        #print(startX, startY, endX, endY, landmarks)
        startX = int(startX)
        startY = int(startY)
        endX = int(endX)
        endY = int(endY)
        bw = endX - startX
        bh = endY - startY
        sx, sy, ex, ey = faceDet.landmark2mouth(landmarks, bw, bh)
        face_img = frame[sy:ey, sx:ex]
        stime = time.time()
        features = featureExtractor.feature_extract(face_img)
        fea_logs.append(time.time() - stime)
        fea_time = 0
        for i in range(len(fea_logs)):
            fea_time += fea_logs[i]
        fea_time = fea_time / len(fea_logs)
        if len(fea_logs) >= TIME_PERIOD:
            fea_logs.pop()
    
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
            stime = time.time()
            pred_in = fin[np.newaxis, :]
            #pred_in = fin
            #print(pred_in.shape)
            pred = model.predict(pred_in)
            pred_logs.append(time.time() - stime)
            pred_time = 0
            for i in range(len(pred_logs)):
                pred_time += pred_logs[i]
            pred_time = pred_time / len(pred_logs)
            if len(pred_logs) >= TIME_PERIOD:
                pred_logs.pop()
            
            try:
                text='pred%.3f'%(1/pred_time)
            except ZeroDivisionError:
                text = 'Inf'
            cv2.putText(frame, text, (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.putText(frame, 'yawn: %d'%(int(pred+0.5)), (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.rectangle(frame, (sx, sy), (ex, ey), (255, 0, 0), 2)
#            idx = frames - 1
#            cv2.rectangle(frame, (cord['sx'][idx], cord['sy'][idx]),
#                          (cord['ex'][idx], cord['ey'][idx]), (0, 255, 0), 2)
        
        cv2.imshow('', frame)
        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            break
    
    vstop.value = 1
    vin.release()
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