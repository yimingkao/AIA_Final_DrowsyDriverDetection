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
#choose 1000 pics to train
num0 = [i for i in range(degrees[0])]
random.shuffle(num0)
num0 = num0[0:1000]
num0.sort()
num5 = [i for i in range(degrees[5])]
random.shuffle(num5)
num5 = num5[0:1000]
num5.sort()
num0[999] = 0
num5[999] = 0

#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html
#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

for i in range(6):
    dst_path = 'deg%d/'%i
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)

idx_0 = 0 # current index of num0
idx_5 = 0 # current index of num5
cnt_0 = 0 # current counter of degree0
cnt_5 = 0 # current counter of degree5
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
        dst_path = 'deg'+str(degree)+'/'
        cv2.imwrite(dst_path+fname.replace('.csv', '_%d.jpg'%i), face_img)
        print('\r%d'%i, end='')
    #break
    
