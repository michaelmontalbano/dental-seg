# extractor.py
import numpy as np
import torch
import cv2
from ultralytics import YOLO
from torchvision import transforms

from inference.features_glcm import extract_glcm_features
from inference.features_gabor import extract_gabor_features
from inference.fractal import fractal_dimension

import mahotas


def compute_geometry(feature_box, implant_box):
    fx1, fy1, fx2, fy2 = feature_box
    ix1, iy1, ix2, iy2 = implant_box

    f_w, f_h = fx2 - fx1, fy2 - fy1
    f_area = f_w * f_h
    f_cx, f_cy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
    i_w, i_h = ix2 - ix1, iy2 - iy1
    rel_x = (f_cx - ix1) / i_w
    rel_y = (f_cy - iy1) / i_h

    return {
        "width": f_w,
        "height": f_h,
        "area": f_area,
        "aspect_ratio": f_w / f_h if f_h > 0 else 0,
        "centroid": (f_cx, f_cy),
        "relative_centroid": (rel_x, rel_y)
    }


class ImplantFeatureExtractor:
    def __init__(self, implant_model_path=None, feature_model_path=None, use_deep=False):
        self.implant_yolo = YOLO(implant_model_path) if implant_model_path else None
        self.feature_yolo = YOLO(feature_model_path)

        self.use_deep = use_deep
        self.deep_model = None  # Not available due to install constraints

    def extract_features(self, image):
        if self.implant_yolo is None:
            return self._extract_features_from_image(image)
        else:
            return self._extract_features_per_implant(image)

    def _extract_features_per_implant(self, image):
        implants = self.implant_yolo(image)[0].boxes.xyxy.cpu().numpy()
        all_features = []
            

        for bbox in implants:
            x1, y1, x2, y2 = map(int, bbox)
            implant_crop = image[y1:y2, x1:x2]

            feature_detections = self.feature_yolo(implant_crop)[0].boxes
            features = []
            feature_classes = feature_detections.cls.cpu().numpy() if hasattr(feature_detections, 'cls') else [None] * len(feature_detections)
            for fbox, fcls in zip(feature_detections.xyxy.cpu().numpy(), feature_classes):
                fx1, fy1, fx2, fy2 = map(int, fbox)
                f_crop = implant_crop[fy1:fy2, fx1:fx2]

                if f_crop.size == 0 or f_crop.shape[0] == 0 or f_crop.shape[1] == 0:
                    continue  # skip invalid crops

                try:
                    geo_feat = self.extract_geometric_features(f_crop)
                    tex_feat = self.extract_texture_features(f_crop)
                    deep_feat = np.array([])  # Deep model disabled
                    stats = compute_geometry((fx1+x1, fy1+y1, fx2+x1, fy2+y1), (x1, y1, x2, y2))

                    features.append({
                        "feature_box": [fx1+x1, fy1+y1, fx2+x1, fy2+y1],
                        "geometry": stats,
                        "class_id": int(fcls) if fcls is not None else None,
                        "features": np.concatenate([geo_feat, tex_feat, deep_feat]).tolist()
                    })
                except Exception as e:
                    print(f"Skipping feature due to error: {e}")
                    continue

            all_features.append({
                "implant_box": bbox.tolist(),
                "features": features
            })

        self._visualize_results(image, implants, all_features)
        return all_features

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

    def _visualize_results(self, image, implant_boxes, all_features):
        vis = image.copy()
        for imp_box, feat_group in zip(implant_boxes, all_features):
            x1, y1, x2, y2 = map(int, imp_box)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for f in feat_group['features']:
                fx1, fy1, fx2, fy2 = map(int, f['feature_box'])
                color = (0, 0, 255) if f.get("class_id") is None else (255, 0, 0)
                cv2.rectangle(vis, (fx1, fy1), (fx2, fy2), color, 1)
        cv2.imshow("Detected Features", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _extract_features_from_image(self, image):
        full_h, full_w = image.shape[:2]
        feature_detections = self.feature_yolo(image)[0].boxes
        feature_classes = feature_detections.cls.cpu().numpy() if hasattr(feature_detections, 'cls') else [None] * len(feature_detections)

        features = []
        for fbox, fcls in zip(feature_detections.xyxy.cpu().numpy(), feature_classes):
            fx1, fy1, fx2, fy2 = map(int, fbox)
            f_crop = image[fy1:fy2, fx1:fx2]

            if f_crop.size == 0 or f_crop.shape[0] == 0 or f_crop.shape[1] == 0:
                continue

            try:
                geo_feat = self.extract_geometric_features(f_crop)
                tex_feat = self.extract_texture_features(f_crop)
                deep_feat = np.array([])
                stats = compute_geometry((fx1, fy1, fx2, fy2), (0, 0, full_w, full_h))

                features.append({
                    "feature_box": [fx1, fy1, fx2, fy2],
                    "geometry": stats,
                    "class_id": int(fcls) if fcls is not None else None,
                    "features": np.concatenate([geo_feat, tex_feat, deep_feat]).tolist()
                })
            except Exception as e:
                print(f"Skipping feature due to error: {e}")
                continue

        self._visualize_results(image, [(0, 0, full_w, full_h)], [{"features": features}])
        return [{"implant_box": [0, 0, full_w, full_h], "features": features}]

    def extract_deep_features(self, roi):
        return np.array([])  # No-op since deep model isn't available
