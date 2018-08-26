
import cv2
import time
import numpy as np
import pandas as pd
import os
import re
import pickle
from keras.models import load_model
import itertools
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



from mobilenet_custom_fea_extract import MobileNetCustomFeatureExtract
from mtcnn_face_det import MTCNNFaceDet

def plot_confusion_matrix(cm, classes, filename,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(filename)
    plt.clf()

window_size = 14
N_FEATURES = 512
featureExtractor = MobileNetCustomFeatureExtract(N_FEATURES)
model = load_model('mobilecus_'+str(N_FEATURES)+'_train.h5')


video_path = '/home/jovyan/at072-group04/ymkao/AIA_Final_DrowsyDriverDetection/model2/Alex_DDD/video/'
marker_path = '/home/jovyan/at072-group04/ymkao/AIA_Final_DrowsyDriverDetection/model2/Alex_DDD/label/'
cord_path = '/home/jovyan/at072-group04/ymkao/AIA_Final_DrowsyDriverDetection/model2/Alex_DDD/bbox/'
faceDet = MTCNNFaceDet()


result = []
files = pd.read_csv('alex.csv')
file_list = files['Name'].values
for fname in file_list:
    golden = []
    answer = []
    frames = 0
    fin = np.empty(shape=(window_size,N_FEATURES))
    vin = cv2.VideoCapture(video_path+fname)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vout = cv2.VideoWriter(fname.replace('.avi', '_out.avi'), fourcc, 30.0, (640,480))
    length = int(vin.get(cv2.CAP_PROP_FRAME_COUNT))
    print('{}: {}'.format(fname, length))
    cord = pd.read_csv(cord_path+fname.replace('.avi', '.csv'))
    marker = pd.read_csv(marker_path+fname.replace('.avi', '.csv'))
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
                cv2.putText(frame, 'yawn: %d'%(int(marker['yawn'][i])), (20, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,), 2)
                cv2.rectangle(frame, (fsx, fsy), (fex, fey), (0, 0, 255), 2)
                cv2.rectangle(frame, (sx, sy), (ex, ey), (255, 0, 0), 2)
                golden.append(int(marker['yawn'][i]))
                answer.append(int(pred+0.5))
        else:
            cv2.putText(frame, 'yawn: ?', (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
        vout.write(frame)
        print('\r%d'%i, end='')
    vin.release()
    vout.release()
    result.append([fname, golden, answer])
    #break
cm = confusion_matrix(golden, answer, labels=[0,1,2,3,4,5])
plot_confusion_matrix(cm, [0,1,2,3,4,5], 'alexddd.png')
with open('alexddd_result.pickle', 'wb') as f:
    pickle.dump(result, f)