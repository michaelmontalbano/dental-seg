from skimage.filters import gabor
import numpy as np

def extract_gabor_features(image):
    responses = []
    for theta in [0, 45, 90, 135]:
        filt_real, _ = gabor(image, frequency=0.6, theta=np.radians(theta))
        responses.extend([filt_real.mean(), filt_real.std()])
    return np.array(responses)
