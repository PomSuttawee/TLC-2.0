import os
import cv2
from typing import List
import numpy as np

def load_images_from_folder(folder_path: str) -> List[np.ndarray]:
    """
    Load all image files from a folder using OpenCV.

    Args:
        folder_path (str): Path to the folder containing image files.

    Returns:
        List[np.ndarray]: List of images loaded as OpenCV BGR NumPy arrays.
    """
    supported_ext = ('.jpg', '.jpeg', '.png')
    images = []

    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(supported_ext):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
            else:
                print(f"Warning: Failed to load {img_path}")
    return images
