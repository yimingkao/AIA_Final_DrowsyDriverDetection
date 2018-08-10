
import cv2
import time

from ssd_face_det import SSDFaceDet
#from dense121_fea_extract import Dense121FeatureExtract

N_FEATURES = 2048
faceDet = SSDFaceDet()
faceDet.reset()
#featureExtracter = Dense121FeatureExtract(N_FEATURES)

vin = cv2.VideoCapture(0)
while True:
    ret, frame = vin.read()
    stime = time.time()
    startX, startY, endX, endY = faceDet.faceFind(frame)
    face_time = time.time() - stime
    #cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

    # Extract features by good model
    startX = int(startX)
    startY = int(startY)
    endX = int(endX)
    endY = int(endY)
    # y: 6/12 -> 11/12, use 5/12 region size
    yoff = int((endY-startY)*6/12)
    xoff = int((endX-startX)/4)
    ybot = int((endY-startY)*1/12)
    face_img = frame[startY+yoff:endY-ybot, startX+xoff:endX]
    #features = featureExtracter.feature_extract(face_img)
    fea_time = time.time() - face_time

    # prediction
    
    text='face%.3f feature%.3f'%(1/face_time, 1/fea_time)
    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
    cv2.rectangle(frame, (startX+xoff, startY+yoff), (endX, endY-ybot), (255, 0, 0), 2)
    
    cv2.imshow(text, frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

vin.release()
