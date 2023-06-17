import os
import cv2

def load_image(images_dir, image_name, channels):
    img_path = os.path.join(images_dir, image_name)
    img = None
    if os.path.isfile(img_path):
        if channels == 3:
            img = cv2.imread(img_path)
            return img
        if channels == 1:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_norm = cv2.normalize(img, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            return img_norm