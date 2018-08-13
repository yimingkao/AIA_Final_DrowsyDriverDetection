
from dense121_fea_extract import Dense121FeatureExtract
from mobilenet_fea_extract import MobileNetFeatureExtract

extractor = 'dense121'
featureExtracter = Dense121FeatureExtract(N_FEATURES)
#extractor = 'mobilenet'
#featureExtracter = MobileNetFeatureExtract(N_FEATURES)

N_FEATURES = 2048
video_path = '/projectdata/driver/YawDD/'
#video_path = '../../../../../YawDD/'

