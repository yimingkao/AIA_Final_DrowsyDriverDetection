from keras.models import load_model
import pandas as pd
import numpy as np
import sys
import cv2
import re
from sklearn.metrics import confusion_matrix

from ymutils import lookup_model
from ymutils import plot_confusion_matrix


def isMale(name):
    return re.match('[0-9]{1,2}-Male.+', name) != None

def preprocess_input(x):
    img = x[:,:,::-1] # opencv BGR -> RGB
    #img = model_dict['preproc'](img)
    img = model_dict['preproc'](img.astype(float))
    #return img[:,:,::-1] # RGB -> opencv BGR # shouldn't do this!!!
    return img

model_dict = ''

def model_inference(model):
    global model_dict
    outpath = 'test_yawn/'
    model_dict = lookup_model(model)
    model_name = model+'_trained.h5'
    shape_used = model_dict['shape']
    model = load_model('saved_models/'+model_name)
    df = pd.read_csv('yawn_test.csv')
    for index, row in df.iterrows():
        path=''
        if isMale(row['Name']):
            path += 'Male/'
        else:
            path += 'Female/'
        main_name = re.match(r'(.*).avi', row['Name']).group(1)
        src_name = path + row['Name']
        dst_name = outpath + main_name + '_test.avi'
        mark_name = path + main_name + '_mark.txt'
        png_name = outpath + main_name + '_cm.png'
        vin = cv2.VideoCapture(src_name)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        vout = cv2.VideoWriter(dst_name, fourcc, 30.0, (640,480))
        length = int(vin.get(cv2.CAP_PROP_FRAME_COUNT))
        print('{}: {}'.format(src_name, length))
        fmark = open(mark_name, 'r')
        golden = []
        answer = []
        for i in range(length):
            ret, frame = vin.read()
            degree = fmark.readline()
            degree = int(degree)
            golden.append(degree)
            #scale frame to shape used and predict by model
            img = cv2.resize(frame, shape_used, interpolation=cv2.INTER_AREA)
            img = preprocess_input(img)
            x_test = np.empty(shape=(1, shape_used[0], shape_used[1], 3))
            x_test[0,:,:,:] = img
            y_pred = model.predict(x_test)
            y_pred = np.argmax(y_pred, axis=1)
            answer.append(y_pred[0])

            if y_pred[0] != degree:
                cv2.putText(frame, str(degree), (20, 440), cv2.FONT_HERSHEY_DUPLEX,
                            2, (0, 0, 255), 1, cv2.LINE_AA)
                
            cv2.putText(frame, str(y_pred[0]), (20, 100), cv2.FONT_HERSHEY_DUPLEX,
                        2, (255, 0, 0), 1, cv2.LINE_AA)
            vout.write(frame)
            print('%d\r'%i, end='')
        print('\n')
        fmark.close()
        vin.release()
        vout.release()
        cm = confusion_matrix(golden, answer, labels=[0,1,2,3,4,5])
        plot_confusion_matrix(cm, [0,1,2,3,4,5], png_name)
        correct = 0
        for i in range(len(golden)):
            if answer[i] == golden[i]:
                correct += 1
        print(main_name + ': %.4f'%(correct / len(golden)*100))
        break


#def model_inference2(model):
#    answer = [0,0,0,0,0]
#    golden = [4,4,4,4,4]
#    cm = confusion_matrix(golden, answer, labels=[0,1,2,3,4,5])
#    plot_confusion_matrix(cm, [0,1,2,3,4,5], 'test.png')

def main():
    if len(sys.argv) < 2:
        print('Usage: ' + sys.argv[0] + ' model')
        print('Support models: ResNet50, inception_v3, xception, inception_resnet_v2, densenet201')
        sys.exit(1)
    model = sys.argv[1]
#    model = 'ResNet50'
    model_inference(model)
    return 0

if __name__ == "__main__":
    main()
