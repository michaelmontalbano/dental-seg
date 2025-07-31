import numpy as np

def fractal_dimension(image):
    def box_count(img, box_size):
        h, w = img.shape
        count = 0
        for i in range(0, h, box_size):
            for j in range(0, w, box_size):
                if np.any(img[i:i+box_size, j:j+box_size]):
                    count += 1
        return count

    img_bin = image > 0.5
    sizes = [2, 4, 8, 16, 32]
    counts = [box_count(img_bin, s) for s in sizes]
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]
