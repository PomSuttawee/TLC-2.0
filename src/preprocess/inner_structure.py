"""

"""

import cv2
import numpy as np

def process_tlc_paper(image: np.ndarray) -> np.ndarray:
    pass

def _detect_reference_lines(image: np.ndarray):
    IMAGE_SHAPE = image.shape
    def _detect_all_lines() -> list:
        """
        Detects lines in the image using the Line Segment Detector (LSD) algorithm.

        Args:
            image (np.ndarray): Input BGR image.

        Returns:
            list: List of detected lines, where each line is represented as a tuple of coordinates.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)
        lines, _, _, _ = lsd.detect(enhanced)
        
        if lines is not None:
            return [line[0] for line in lines]
        return None
    
    def _filter_lines(lines: list) -> list:
        """
        Filters lines based on their position.

        Args:
            lines (list): List of detected lines.

        Returns:
            list: Filtered list of lines.
        """
        valid_position = [(0, 0.15 * IMAGE_SHAPE[0]), (0.85 * IMAGE_SHAPE[0], IMAGE_SHAPE[0])]
        return [*map(lambda x: x if (valid_position[0][0] <= x <= valid_position[0][1]) or (valid_position[1][0] <= x <= valid_position[1][1]) else None)]
    
    lines = _detect_all_lines()
    filtered_lines = _filter_lines(lines)

