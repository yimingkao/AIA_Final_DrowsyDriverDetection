import cv2
import pandas as pd
import re

def isMale(name):
    return re.match('[0-9]{1,2}-Male.+', name) != None

set_name = 'yawn_train'
video_path = ''
faceroi_path = 'ssd_face/bbox/'
set_path = set_name+'/'

data = pd.read_csv(video_path+'yawn_train.csv')
target_path = '/'+video_path+faceroi_path+set_path
print(data)
print(target_path)
for i in range(len(data)):
    filename = target_path + data['Name'][i]
    filename1 = '/at072-group04/model2/YawDD/ssd_face/roi/'+data['Name'][i]
    print(filename1)