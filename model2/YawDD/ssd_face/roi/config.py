
#from dense121_fea_extract import Dense121FeatureExtract
#from mobilenet_fea_extract import MobileNetFeatureExtract
from mobilenet_custom_fea_extract import MobileNetCustomFeatureExtract

N_FEATURES = 512 
video_path = '/projectdata/driver/YawDD/'
#video_path = '../../../../../YawDD/'

#extractor = 'dense121'
#featureExtracter = Dense121FeatureExtract(N_FEATURES)
#extractor = 'mobilenet'
#featureExtracter = MobileNetFeatureExtract(N_FEATURES)
extractor = 'mobilecus'
featureExtracter = MobileNetCustomFeatureExtract(N_FEATURES)
