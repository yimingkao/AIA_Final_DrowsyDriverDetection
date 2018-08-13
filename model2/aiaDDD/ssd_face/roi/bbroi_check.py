
import cv2
import pandas as pd
import os

dataset_path = '../../../../../aiaDDD/'
video_path = dataset_path + 'videos/'

for filelst in ['train.csv', 'test.csv']:
    dst_path = filelst.replace('.csv','/')
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    data = pd.read_csv(dataset_path+filelst)
    for i in range(len(data)):
        filename = video_path + data['Name'][i]
        vin = cv2.VideoCapture(filename)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        vout = cv2.VideoWriter(dst_path+data['Name'][i], fourcc, 30.0, (640,360))
        length = int(vin.get(cv2.CAP_PROP_FRAME_COUNT))
        print('{}: {}'.format(filename, length))
        mark = pd.read_csv(filename.replace('.avi', '.csv'))
        cord = pd.read_csv('../bbox/'+data['Name'][i].replace('.avi', '_bbox.csv'))    
        for j in range(length):
            ret, frame = vin.read()
            startX = cord['sx'][j] if cord['sx'][j] >= 0 else 0
            startY = cord['sy'][j] if cord['sy'][j] >= 0 else 0
            endX = cord['ex'][j]
            endY = cord['ey'][j]
            # y: 6/12 -> 11/12, use 5/12 region size
            yoff = int((endY-startY)*6/12)
            xoff = int((endX-startX)/4)
            ybot = int((endY-startY)*1/12)
            
            cv2.rectangle(frame, (startX+xoff, startY+yoff), (endX, endY-ybot),
                    (0, 0, 255), 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                    (255, 0, 0), 2)
            
            cv2.putText(frame, 'yawn: %d'%mark['yawn'][j], (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, 'ec  : %d'%mark['ec'][j], (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, 'nod : %d'%mark['nod'][j], (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  
            
            vout.write(frame)
            print('\r%d'%j,end='')
            
        vout.release()
        vin.release()
        #break