import os
import cv2

INGREDIENT_FOLDER = os.path.join(os.path.dirname(__file__), 'data\\ingredients')
MIXTURE_FOLDER = os.path.join(os.path.dirname(__file__), 'data\\mixtures')

def _read_image_from_directory(directory) -> dict:
    filenames = [os.path.join(directory, name) for name in os.listdir(directory)]
    return {filename : cv2.imread(filename) for filename in filenames}

def get_test_image() -> dict:
    images = _read_image_from_directory(INGREDIENT_FOLDER)
    images.update(_read_image_from_directory(MIXTURE_FOLDER))
    return images