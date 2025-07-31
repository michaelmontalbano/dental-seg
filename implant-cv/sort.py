import os
import cv2
import shutil

# ----------- Paths -----------
IMAGE_DIR = "/Users/michaelmontalbano/work/repos/cej-ac-detection/dir/inference/pipeline/imp11-12/train/images"
TO_DELETE_DIR = os.path.join(IMAGE_DIR, "to_delete")
os.makedirs(TO_DELETE_DIR, exist_ok=True)

# ----------- Load Images -----------
images = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
    and os.path.isfile(os.path.join(IMAGE_DIR, f))
])

i = 0
while i < len(images):
    img_path = os.path.join(IMAGE_DIR, images[i])
    img = cv2.imread(img_path)

    if img is None:
        print(f"Could not load {images[i]}, skipping...")
        i += 1
        continue

    cv2.imshow(f"{i+1}/{len(images)}: {images[i]} (Right=Next, d=Delete, q=Quit)", img)
    key = cv2.waitKey(0)

    if key == ord('q'):
        break
    elif key == ord('d'):
        # Delete image (move to to_delete)
        target_path = os.path.join(TO_DELETE_DIR, images[i])
        shutil.move(img_path, target_path)
        print(f"Moved to delete: {images[i]}")
        images.pop(i)
        continue
    else:
        i += 1

cv2.destroyAllWindows()
