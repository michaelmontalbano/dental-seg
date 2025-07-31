import numpy as np
import torch
import cv2
from ultralytics import YOLO
from torchvision import transforms

from inference.features_glcm import extract_glcm_features
from inference.features_gabor import extract_gabor_features
from inference.fractal import fractal_dimension
from inference.deep_models import CNNTransformerHybrid

import mahotas

class ImplantFeatureExtractor:
    def __init__(self, implant_model_path, feature_model_path, use_deep=True):
        self.implant_yolo = YOLO(implant_model_path)
        self.feature_yolo = YOLO(feature_model_path)
        self.use_deep = use_deep
        if self.use_deep:
            self.deep_model = CNNTransformerHybrid(num_classes=5)
            self.deep_model.eval()

    def extract_features(self, image):
        implants = self.implant_yolo(image)[0].boxes.xyxy.cpu().numpy()
        all_features = []

        for bbox in implants:
            x1, y1, x2, y2 = map(int, bbox)
            crop = image[y1:y2, x1:x2]
            features = self._analyze_implant_region(crop)
            all_features.append({
                "implant_box": bbox.tolist(),
                "features": features.tolist()
            })

        return all_features

    def _analyze_implant_region(self, roi):
        geo_feat = self.extract_geometric_features(roi)
        tex_feat = self.extract_texture_features(roi)
        deep_feat = self.extract_deep_features(roi) if self.use_deep else np.array([])

        return np.concatenate([geo_feat, tex_feat, deep_feat])

    def extract_geometric_features(self, roi):
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_resized = cv2.resize(roi_gray, (128, 128))
        zernike = mahotas.features.zernike_moments(roi_resized, radius=64)
        fractal = [fractal_dimension(roi_resized)]
        return np.concatenate([zernike, fractal])

    def extract_texture_features(self, roi):
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_resized = cv2.resize(roi_gray, (128, 128))
        glcm = extract_glcm_features(roi_resized)
        gabor = extract_gabor_features(roi_resized)
        return np.concatenate([glcm, gabor])

    def extract_deep_features(self, roi):
        img_resized = cv2.resize(roi, (224, 224))
        tensor = transforms.ToTensor()(img_resized).unsqueeze(0)
        with torch.no_grad():
            return self.deep_model(tensor).cpu().numpy().flatten()


