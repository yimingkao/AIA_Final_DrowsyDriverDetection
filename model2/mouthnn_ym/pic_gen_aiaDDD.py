import pandas as pd
import os
import random
import cv2
from mtcnn_face_det import MTCNNFaceDet

faceDet = MTCNNFaceDet()

excludes = ['VID_20180804_154947.csv',
            'VID_20180804_155059.csv',
            'VID_20180804_155217.csv',
            'VID_20180804_155334.csv',
            'VID_20180804_155437.csv']

mark_path = '../../aiaDDD/markers/'
video_path = '/home/jovyan/at072-group04/aiaDDD/videos/'
bbox_path = '../aiaDDD/mtcnn_face/bbox/'

degree_list = []
for root,subdir,files in os.walk(mark_path):
    for file in files:
        if file in excludes:
            continue
        df = pd.read_csv(mark_path+file)
        counts = [file]
        for i in range(6):
            counts.append(len(df[df['yawn'] == i]['yawn'].values))
            #print(len(df[df['yawn'] == i]['yawn'].values))
        degree_list.append(counts)

#print(degree_list)
degrees = []
for i in range(6):
    total = 0
    for j in range(len(degree_list)):
        total += degree_list[j][i+1]
    print('degree %d %d'%(i, total))
    degrees.append(total)
print(degrees)

#degree 0 79888
#degree 1 1009
#degree 2 978
#degree 3 980
#degree 4 943
#degree 5 6341

for entry in degree_list:
    faceDet.reset()
    fname = entry[0]
    vin = cv2.VideoCapture(video_path + fname.replace('.csv', '.avi'))
    length = int(vin.get(cv2.CAP_PROP_FRAME_COUNT))
    print('{}: {}'.format(fname, length))
    marker = pd.read_csv(mark_path+fname)
    bbox = pd.read_csv(bbox_path+fname)
    for i in range(length):
        ret, frame = vin.read()
        degree = int(marker['yawn'][i]+0.1)
        if degree == 0:
            dst_path = '0/'
        elif degree == 5 or degree == 4:
            dst_path = '1/'
        else:
            continue
        line = bbox.iloc[j].values
        sx = int(line[1])
        sy = int(line[2])
        ex = int(line[3])
        ey = int(line[4])
        bw = ex - sx
        bh = ey - sy
        if bw == 0 or bh == 0:
            continue
        line = line[5:]
        sx, sy, ex, ey = faceDet.landmark2mouth(line, bw, bh)
        face_img = frame[sy:ey, sx:ex]
        cv2.imwrite(dst_path+fname.replace('.csv', '_%d.jpg'%i), face_img)
        print('\r%d'%i, end='')
    #break
    
