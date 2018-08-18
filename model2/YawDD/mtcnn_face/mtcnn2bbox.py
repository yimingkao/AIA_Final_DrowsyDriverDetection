
import cv2
import pandas as pd
import re

from mtcnn_face_det import MTCNNFaceDet


def isMale(name):
    return re.match('[0-9]{1,2}-Male.+', name) != None

video_path = '/projectdata/driver/YawDD/'
faceDet = MTCNNFaceDet()

for set_name in ['yawn_train', 'yawn_valid', 'yawn_test']:
    data = pd.read_csv('../'+set_name+'.csv')
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
        dstname = 'bbox/'+data['Name'][i].replace('.avi', '.csv')
        fout = open(dstname, 'w')
        ostr = 'frame,sx,sy,ex,ey'
        for l in range(5):
            ostr += ',lx%d,ly%x'%(l, l)
        fout.write(ostr+'\n')
        faceDet.reset()
        for j in range(length):
            ret, frame = vin.read()
            startX, startY, endX, endY, lms = faceDet.faceFind(frame)
            ostr = '%d,%d,%d,%d,%d'%(j,startX,startY,endX,endY)
            for l in range(5):
                ostr += ','+str(lms[0])+','+str(lms[1])
                lms = lms[2:]
            fout.write(ostr+'\n')
            print('\r%d'%j,end='')            
        vin.release()
        fout.close()
        #break
    #break
