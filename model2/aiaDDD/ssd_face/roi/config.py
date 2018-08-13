
from dense121_fea_extract import Dense121FeatureExtract
from mobilenet_fea_extract import MobileNetFeatureExtract

N_FEATURES = 1024
video_path = '/projectdata/driver/YawDD/'
#video_path = '../../../../../YawDD/'

extractor = 'dense121'
featureExtracter = Dense121FeatureExtract(N_FEATURES)
#extractor = 'mobilenet'
#featureExtracter = MobileNetFeatureExtract(N_FEATURES)

