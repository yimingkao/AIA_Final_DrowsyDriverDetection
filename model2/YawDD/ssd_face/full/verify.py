
import cv2
import pandas as pd
import re
#shape_used=(224, 224)
def isMale(name):
    return re.match('[0-9]{1,2}-Male.+', name) != None

set_name = 'yawn_test'
video_path = ''
set_path = set_name+'/'

data = pd.read_csv('../../yawn_test.csv')
for i in range(len(data)):
    target_path = '/projectdata/driver/YawDD/'
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
    cord = pd.read_csv('../bbox/'+set_path+data['Name'][i].replace('.avi', '.csv'))    
    for j in range(length):
        ret, frame = vin.read()
        startX = cord['sx'][j] if cord['sx'][j] >= 0 else 0
        startY = cord['sy'][j] if cord['sy'][j] >= 0 else 0
        endX = cord['ex'][j]
        endY = cord['ey'][j]
        yoff = int((endY-startY)*2/3)
        xoff = int((endX-startX)/4)        
        # y: 6/12 -> 11/12, use 5/12 region size
        yoff = int((endY-startY)*6/12)
        xoff = int((endX-startX)/4)
        ybot = int((endY-startY)*1/12)
        
        cv2.rectangle(frame, (startX+xoff, startY+yoff), (endX, endY-ybot),
                (0, 0, 255), 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY),
                (255, 0, 0), 2)
        vout.write(frame)
        print('\r%d'%j,end='')
        
    vout.release()
    vin.release()
    #break