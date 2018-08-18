import pandas as pd
import os
import random
import cv2
import time
import re

def isMale(name):
    return re.match('[0-9]{1,2}-Male.+', name) != None

mark_path = '../../YawDD/markers/'
video_path = '/home/jovyan/projectdata/driver/YawDD/'
bbox_path = '../YawDD/ssd_face/bbox/yawn_train/'

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
#choose 1250 pics to train
num0 = [i for i in range(degrees[0])]
random.shuffle(num0)
num0 = num0[0:1250]
num0.sort()
num5 = [i for i in range(degrees[5])]
random.shuffle(num5)
num5 = num5[0:1250]
num5.sort()
num0[1249] = 0
num5[1249] = 0

##https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html
#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#
for i in range(6):
    dst_path = 'deg%d/'%i
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)

idx_0 = 0 # current index of num0
idx_5 = 0 # current index of num5
cnt_0 = 0 # current counter of degree0
cnt_5 = 0 # current counter of degree5
for entry in degree_list:
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
        # Extract features by good model
        startX = int(bbox['sx'][i])
        startY = int(bbox['sy'][i])
        endX = int(bbox['ex'][i])
        endY = int(bbox['ey'][i])
        # y: 6/12 -> 11/12, use 5/12 region size
        yoff = int((endY-startY)*6/12)
        xoff = int((endX-startX)/4)
        ybot = int((endY-startY)*1/12)
        face_img = frame[startY+yoff:endY-ybot, startX+xoff:endX]
        #face_img = cv2.resize(face_img, (100, 100))
        #face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        #face_img = clahe.apply(face_img)
        degree = int(marker['yawn'][i]+0.1)
        dst_path = 'deg'+str(degree)+'/'
        if degree == 0:
            if num0[idx_0] == cnt_0:
                idx_0 += 1
                cnt_0 += 1
            else:
                cnt_0 += 1
                continue
        elif degree == 5:
            if num5[idx_5] == cnt_5:
                idx_5 += 1
                cnt_5 += 1
            else:
                cnt_5 += 1
                continue            
        cv2.imwrite(dst_path+fname.replace('.avi', '_%d.jpg'%i), face_img)
        print('\r%d'%i, end='')
    #break
    
