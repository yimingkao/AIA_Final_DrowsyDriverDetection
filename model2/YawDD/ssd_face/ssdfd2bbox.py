
import cv2
import pandas as pd
import re

from ssd_face_det import SSDFaceDet


def isMale(name):
    return re.match('[0-9]{1,2}-Male.+', name) != None

video_path = '/projectdata/driver/YawDD/'
faceDet = SSDFaceDet()

data = pd.read_csv('../yawn_train.csv')
for i in range(len(data)):
    target_path = video_path
    if isMale(data['Name'][i]):
        target_path += 'Male/'
    else:
        target_path += 'Female/'
    filename = target_path + data['Name'][i]
    vin = cv2.VideoCapture(filename)
    length = int(vin.get(cv2.CAP_PROP_FRAME_COUNT))
    print('{}: {}'.format(filename, length))
    dstname = data['Name'][i].replace('.avi', '.csv')
    fout = open(dstname, 'w')
    fout.write('frame,sx,sy,ex,ey\n')
    faceDet.reset()
    for j in range(length):
        ret, frame = vin.read()
        startX, startY, endX, endY = faceDet.faceFind(frame)
        fout.write('%d,%d,%d,%d,%d\n'%(j,startX,startY,endX,endY))
        print('\r%d'%j,end='')
        
    vin.release()
    fout.close()
    #break