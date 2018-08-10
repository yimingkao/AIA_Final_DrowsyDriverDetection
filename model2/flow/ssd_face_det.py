
import cv2
import numpy as np

class SSDFaceDet(object):
    def __init__(self):
        self.net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 
                        'res10_300x300_ssd_iter_140000.caffemodel')
        self.reset()
        
    def reset(self):
        (self.startX, self.startY, self.endX, self.endY) = (0, 0, 0, 0)

    def faceFind(self, frame):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        if detections.shape[2] != 0:
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence < 0.5:
                    continue
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (self.startX, self.startY, self.endX, self.endY) = box.astype("int")
                break # one face is enough
        else:
            print('using old position %d %d %d %d\n'%(self.startX, self.startY,
                                                      self.endX, self.endY))
        return self.startX, self.startY, self.endX, self.endY
