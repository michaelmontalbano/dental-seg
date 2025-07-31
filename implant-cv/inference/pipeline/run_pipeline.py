# run_pipeline.py

import os, json, argparse, cv2
from inference.extractor import ImplantFeatureExtractor

def load_image(path):
    return cv2.imread(path)

def main(input_dir, output_path, implant_model_path, feature_model_path):
    extractor = ImplantFeatureExtractor(
        implant_model_path=implant_model_path,
        feature_model_path=feature_model_path,
        use_deep=True
    )

    output = {}
    for file in os.listdir(input_dir):
        if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        image_path = os.path.join(input_dir, file)
        image = load_image(image_path)
        results = extractor.extract_features(image)
        output[file] = results

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output', type=str, required=True, dest='output_path')
    parser.add_argument('--feature_model_path', type=str, default='models/feature_model.pt')
    parser.add_argument('--implant_model_path', type=str, default='models/implant_model.pt')
    args = parser.parse_args()
    main(**vars(args))

