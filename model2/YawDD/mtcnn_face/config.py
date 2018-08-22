
#from mobilenet_custom_fea_extract import MobileNetCustomFeatureExtract
from mouthnnym_fea_extract import MouthnnYMFeatureExtract

N_FEATURES = 512
video_path = '/projectdata/driver/YawDD/'

#extractor = 'mobilecus'
#featureExtractor = MobileNetCustomFeatureExtract(N_FEATURES)
extractor = 'mouthnnym'
featureExtractor = MouthnnYMFeatureExtract(N_FEATURES)

