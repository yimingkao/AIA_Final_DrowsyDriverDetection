import pandas as pd
import os
import random
import cv2
import re
from mtcnn_face_det import MTCNNFaceDet

def isMale(name):
    return re.match('[0-9]{1,2}-Male.+', name) != None

faceDet = MTCNNFaceDet()

mark_path = '../../YawDD/markers/'
video_path = '/home/jovyan/projectdata/driver/YawDD/'
bbox_path = '../YawDD/mtcnn_face/bbox/'

degree_list = []

df = pd.read_csv('../YawDD/yawn_train.csv')
file_list = df['Name'].values
for file in file_list:
    df = pd.read_csv(mark_path+file.replace('.avi', '.csv'))
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

#degree 0 30717
#degree 1 1218
#degree 2 1220
#degree 3 1210
#degree 4 1220
#degree 5 6741

for entry in degree_list:
    faceDet.reset()
    fname = entry[0]
    target = video_path
    if isMale(fname):
        target += 'Male/'
    else:
        target += 'Female/'
    vin = cv2.VideoCapture(target + fname)
    length = int(vin.get(cv2.CAP_PROP_FRAME_COUNT))
    print('{}: {}'.format(fname, length))
    marker = pd.read_csv(mark_path+fname.replace('.avi', '.csv'))
    bbox = pd.read_csv(bbox_path+fname.replace('.avi', '.csv'))
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
        cv2.imwrite(dst_path+fname.replace('.avi', '_%d.jpg'%i), face_img)
        print('\r%d'%i, end='')
    #break
    
