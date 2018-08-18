
from mobilenet_custom_fea_extract import MobileNetCustomFeatureExtract

N_FEATURES = 512
video_path = '/projectdata/driver/YawDD/'

extractor = 'mobilecus'
featureExtractor = MobileNetCustomFeatureExtract(N_FEATURES)

