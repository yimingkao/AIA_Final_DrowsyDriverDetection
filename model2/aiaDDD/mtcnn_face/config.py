
#from mobilenet_custom_fea_extract import MobileNetCustomFeatureExtract
from mouthnnym_fea_extract import MouthnnYMFeatureExtract

N_FEATURES = 512
video_path = '/home/jovyan/at072-group04/aiaDDD/videos/'

extractor = 'mouthnnym'
#featureExtractor = MobileNetCustomFeatureExtract(N_FEATURES)
featureExtractor = MouthnnYMFeatureExtract(N_FEATURES)
