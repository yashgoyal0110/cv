"""Step 1: load MRI images and turn each one into a HOG feature vector (paper Sec. 3.2-3.3)."""
import os

import cv2
import numpy as np
from skimage.feature import hog

# Folder names in the dataset. Their position = label number (0, 1, 2, 3), same as the paper.
CLASSES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]


def preprocess(image, use_clahe):
    """Grayscale image -> (optional CLAHE) -> 16x16 -> values in 0-1."""
    if use_clahe:
        # Our extension: boost local contrast at full resolution, before shrinking
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
    image = cv2.resize(image, (16, 16))
    return image / 255.0


def extract_hog(image):
    """HOG with the paper's settings -> 2916 numbers per image."""
    features = hog(image,
                   orientations=9,           # 9 direction bins (0-180 degrees)
                   pixels_per_cell=(2, 2),   # 16x16 image -> 8x8 cells
                   cells_per_block=(3, 3),   # 6x6 blocks of 3x3 cells
                   block_norm="L2-Hys")
    return features / 255.0  # extra scaling step stated in the paper


def load_dataset(folder, use_clahe=False):
    """Read every image in folder/<class_name>/ and return (features, labels)."""
    features, labels = [], []
    for label, class_name in enumerate(CLASSES):
        class_folder = os.path.join(folder, class_name)
        for file_name in sorted(os.listdir(class_folder)):
            image = cv2.imread(os.path.join(class_folder, file_name), cv2.IMREAD_GRAYSCALE)
            if image is None:  # skip files that are not images
                continue
            features.append(extract_hog(preprocess(image, use_clahe)))
            labels.append(label)
    return np.array(features), np.array(labels)
