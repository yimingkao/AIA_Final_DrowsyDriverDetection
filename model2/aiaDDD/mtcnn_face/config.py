
from mobilenet_custom_fea_extrac import MobileNetCustomFeatureExtract

N_FEATURES = 1024
video_path = '/home/jovyan/at072-group04/aiaDDD/videos/'

extractor = 'mobilecus'
featureExtractor = MobileNetCustomFeatureExtract(N_FEATURES)

