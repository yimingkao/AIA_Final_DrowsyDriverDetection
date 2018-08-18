
import cv2
import pandas as pd

from mtcnn_face_det import MTCNNFaceDet


video_path = '/home/jovyan/at072-group04/aiaDDD/videos/'
faceDet = MTCNNFaceDet()

for set_name in ['train', 'test']:
    data = pd.read_csv('../'+set_name+'.csv')
    for i in range(len(data)):
        filename = data['Name'][i]
        vin = cv2.VideoCapture(video_path+filename)
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
            if len(lms):
                for l in range(5):
                    ostr += ','+str(lms[0])+','+str(lms[1])
                    lms = lms[2:]
            else:
                ostr += '0,0,0,0,0,0,0,0,0,0'
            fout.write(ostr+'\n')
            print('\r%d'%j,end='')            
        vin.release()
        fout.close()
        #break
    #break
