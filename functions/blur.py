import cv2, numpy as np


def inspect_quality(path, blur_th=50, dark_th=30):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    bright_score = np.mean(gray)

    is_blur = blur_score < blur_th
    is_dark = bright_score < dark_th
    return int(is_blur or is_dark)
