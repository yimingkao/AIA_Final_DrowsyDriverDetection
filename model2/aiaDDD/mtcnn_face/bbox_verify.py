
import cv2
import pandas as pd

from mtcnn_face_det import MTCNNFaceDet


video_path = '/home/jovyan/at072-group04/aiaDDD/videos/'
#video_path = '../../../../../at072-group04/aiaDDD/videos/'
faceDet = MTCNNFaceDet()

for set_name in ['train', 'test']:
    data = pd.read_csv('../'+set_name+'.csv')
    for i in range(len(data)):
        filename = data['Name'][i]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        vout = cv2.VideoWriter(filename, fourcc, 30.0, (640,360))
        vin = cv2.VideoCapture(video_path+filename)
        length = int(vin.get(cv2.CAP_PROP_FRAME_COUNT))
        print('{}: {}'.format(filename, length))
        dstname = 'bbox/'+data['Name'][i].replace('.avi', '.csv')
        cord = pd.read_csv(dstname)
        for j in range(length):
            ret, frame = vin.read()
            line = cord.iloc[j].values
            cv2.rectangle(frame, (int(line[1]), int(line[2])),
                          (int(line[3]), int(line[4])),
                          (0, 0, 255), 2)
            bw = line[3] - line[1]
            bh = line[4] - line[2]
            line = line[5:]
            if line[6]:
                sx, sy, ex, ey = faceDet.landmark2mouth(line, bw, bh)
                cv2.rectangle(frame, (sx, sy), (ex, ey), (255, 0, 0), 2)
            vout.write(frame)
            print('\r%d'%j,end='')            
        vin.release()
        vout.release()
        #break
    #break
