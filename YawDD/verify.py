
import cv2
import pandas as pd
import re
#shape_used=(224, 224)
def isMale(name):
    return re.match('[0-9]{1,2}-Male.+', name) != None

set_name = 'yawn_train'
video_path = ''
faceroi_path = 'ssd_face/bbox/'
set_path = set_name+'/'

data = pd.read_csv(video_path+'yawn_train.csv')
for i in range(len(data)):
    target_path ='/at072-group04/model2/YawDD/'+video_path+faceroi_path+set_path
    target_path1 ='/projectdata/driver/YawDD/'
    if isMale(data['Name'][i]):
        target_path1 += 'Male/'
    else:
        target_path1 += 'Female/'
    filename = target_path + data['Name'][i]
    filename1 = target_path1+data['Name'][i]
    vin = cv2.VideoCapture(filename1)
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
        yoff = int((endY-startY)*2/3)
        xoff = int((endX-startX)/4)
        #face_img = frame[startY+yoff:endY, startX+xoff:endX]
        #face_img = cv2.resize(face_img, shape_used)
        
        cv2.rectangle(frame, (startX+xoff, startY+yoff), (endX, endY),
                (0, 0, 255), 2)
        cv2.rectangle(frame, (startY, startX), (endX, endY),
                (255, 0, 255), 2)
        vout.write(frame)
        #vout.write(face_img)
        print('\r%d'%j,end='')
        
        
    vout.release()
    vin.release()
    #break