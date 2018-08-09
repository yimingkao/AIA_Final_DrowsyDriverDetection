
import cv2
import pandas as pd
import re

def isMale(name):
    return re.match('[0-9]{1,2}-Male.+', name) != None

set_name = 'yawn_valid'
video_path = '../../YawDD/'
faceroi_path = 'ssd_face/'
set_path = set_name+'/'

data = pd.read_csv(video_path+'yawn_valid.csv')
for i in range(len(data)):
    target_path = video_path+'Mirror/'
    if isMale(data['Name'][i]):
        target_path += 'Male/'
    else:
        target_path += 'Female/'
    filename = target_path + data['Name'][i]
    vin = cv2.VideoCapture(filename)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vout = cv2.VideoWriter(data['Name'][i], fourcc, 30.0, (640,480))
    length = int(vin.get(cv2.CAP_PROP_FRAME_COUNT))
    print('{}: {}'.format(filename, length))
    cord = pd.read_csv(faceroi_path+set_path+data['Name'][i].replace('.avi', '.csv'))    
    for j in range(length):
        ret, frame = vin.read()
        startX = cord['sx'][j] if cord['sx'][j] >= 0 else 0
        startY = cord['sy'][j] if cord['sy'][j] >= 0 else 0
        endX = cord['ex'][j]
        endY = cord['ey'][j]
        cv2.rectangle(frame, (startX, startY), (endX, endY),
                (0, 0, 255), 2)
        vout.write(frame)
        print('\r%d'%j,end='')
        
    vout.release()
    vin.release()
    #break