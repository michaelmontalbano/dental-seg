from skimage.feature import graycomatrix, graycoprops
import numpy as np

def extract_glcm_features(image):
    img = (image * 255).astype(np.uint8)
    img = img // 4  # 64 gray levels
    glcm = graycomatrix(img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)
    features = []
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy']:
        features.extend(graycoprops(glcm, prop).flatten())
    return np.array(features)
