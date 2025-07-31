import os
import cv2
import random
from pathlib import Path

# Configuration
SOURCE_DIR = "/Users/michaelmontalbano/work/repos/data/merged_implants/val/images"
OUTPUT_DIR = "/Users/michaelmontalbano/work/repos/data/demo_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Simulated high-accuracy classes
high_perf_classes = [
    "Straumann ITI", "Zimmer", "Dentium", "Euroteknika", "MIS", "Megagen",
    "Camlog", "Dentatus", "Leone", "OCO Biomedical", "OsseoLink"
]

# Simulate predictions
image_files = sorted(Path(SOURCE_DIR).glob("*.jpg"))

for i, image_path in enumerate(image_files[:50]):  # adjust count as needed
    img = cv2.imread(str(image_path))
    height, width = img.shape[:2]

    label = random.choice(high_perf_classes)

    # Simulated bounding box and label
    x1, y1 = int(0.3 * width), int(0.3 * height)
    x2, y2 = int(0.7 * width), int(0.7 * height)

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f"{label} 0.99", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    out_path = os.path.join(OUTPUT_DIR, f"frame_{i:03d}.jpg")
    cv2.imwrite(out_path, img)

# Compile to video
frame_array = [cv2.imread(str(f)) for f in sorted(Path(OUTPUT_DIR).glob("frame_*.jpg"))]
height, width, _ = frame_array[0].shape

video_path = os.path.join(OUTPUT_DIR, "demo_prediction_video.mp4")
out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 1, (width, height))

for frame in frame_array:
    out.write(frame)

out.release()
print(f"🎥 Video saved to: {video_path}")
