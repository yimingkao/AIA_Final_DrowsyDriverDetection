
import numpy as np
from MTCNN.MtcnnDetector import MtcnnDetector
from MTCNN.detector import Detector
from MTCNN.fcn_detector import FcnDetector
from MTCNN.mtcnn_model import P_Net, R_Net, O_Net

class MTCNNFaceDet(object):
    def __init__(self):
        thresh = [0.9, 0.6, 0.7]
        min_face_size = 150
        stride = 2
        slide_window = False
        detectors = [None, None, None]
        prefix = ['MTCNN/PNet/PNet', 'MTCNN/RNet/RNet', 'MTCNN/ONet/ONet']
        epoch = [18, 14, 16]
        model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
        detectors[0] = FcnDetector(P_Net, model_path[0])
        detectors[1] = Detector(R_Net, 24, 1, model_path[1])
        detectors[2] = Detector(O_Net, 48, 1, model_path[2])
        self.detector = MtcnnDetector(detectors=detectors,
                                      min_face_size=min_face_size,
                                      stride=stride,
                                      threshold=thresh,
                                      slide_window=slide_window)
        self.reset()
        
    def reset(self):
        (self.startX, self.startY, self.endX, self.endY) = (0, 0, 0, 0)
        self.landmarks = []

    def faceFind(self, frame):
        image = np.array(frame)
        boxes, landmarks = self.detector.detect(image)
        if boxes.shape[0]:
            [self.startX, self.startY, self.endX, self.endY] = boxes[0,:4]
            self.landmarks = landmarks[0]
        else:
            print('using old position %d %d %d %d\n'%(self.startX, self.startY,
                                                      self.endX, self.endY))
        return self.startX, self.startY, self.endX, self.endY, self.landmarks

    def landmark2mouth(self, landmark, box_width, box_height):
        mouth_width_left_ratio = 1. / 18.
        mouth_width_right_ratio = 1. / 8.
        mouth_height_up_ratio = 1. / 12.
        mouth_height_down_ratio = 1. / 3.5
        sx = int(landmark[6] - mouth_width_left_ratio * box_width)
        sy = int(landmark[7] - mouth_height_up_ratio * box_height)
        ex = int(landmark[8] + mouth_width_right_ratio * box_width)
        ey = int(landmark[9] + mouth_height_down_ratio * box_height)
        return sx, sy, ex, ey
