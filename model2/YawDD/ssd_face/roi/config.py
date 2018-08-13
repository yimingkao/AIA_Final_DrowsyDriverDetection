
from dense121_fea_extract import Dense121FeatureExtract
from mobilenetv2_fea_extract import MobileNetV2FeatureExtract

extractor = 'dense121'
featureExtracter = Dense121FeatureExtract(N_FEATURES)
#extractor = 'mobilenetv2'
#featureExtracter = MobileNetV2FeatureExtract(N_FEATURES)

N_FEATURES = 2048
video_path = '/projectdata/driver/YawDD/'
#video_path = '../../../../../YawDD/'

