
from mobilenet_custom_fea_extract import MobileNetCustomFeatureExtract

N_FEATURES = 512
video_path = '/home/jovyan/at072-group04/aiaDDD/videos/'

extractor = 'mobilecus'
featureExtractor = MobileNetCustomFeatureExtract(N_FEATURES)

