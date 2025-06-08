"""
This module contains functions to process TLC paper images by removing the background,
cropping to the region of interest, and applying a perspective transform.
It uses OpenCV for image processing tasks.
"""

from typing import Tuple
import numpy as np
import cv2

def crop_plate(image: np.ndarray, crop: bool = True, transform: bool = True) -> np.ndarray:
    """
    Processes a TLC paper image by removing background and optionally cropping and applying perspective transform.

    Args:
        image (np.ndarray): Input BGR image of the TLC paper.
        crop (bool, optional): Whether to crop the image to the region of interest. Defaults to True.
        transform (bool, optional): Whether to apply a perspective transform. Defaults to True.

    Returns:
        np.ndarray: An image after processing steps (background removed, optionally cropped and transformed).
    """
    result, contour = _remove_background(image)
    if crop:
        result = _crop_to_roi(result, contour)
    if transform:
        result = _apply_perspective_transform(result)
    return result

def _remove_background(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Removes the background from the TLC paper image by finding and masking the largest contour.

    Args:
        image (np.ndarray): Input BGR image.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple (cleaned_image, contour), where cleaned_image is the masked result,
        and contour is the largest contour found, or None if none are found.
    """
    max_contour = _find_max_contour(image)
    if max_contour is None:
        return image, None
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [max_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
    return cv2.bitwise_and(image, mask), max_contour

def _crop_to_roi(image: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """
    Crops the image to the bounding rectangle of the given contour.

    Args:
        image (np.ndarray): Input BGR image.
        contour (np.ndarray): Contour defining the region of interest.

    Returns:
        np.ndarray: Cropped image if contour is valid; otherwise the original image.
    """
    if contour is None:
        return image
    x, y, w, h = cv2.boundingRect(contour)
    return image[y:y+h, x:x+w]

def _apply_perspective_transform(image: np.ndarray) -> np.ndarray:
    """
    Applies a perspective transform using the largest contour as a reference.

    Args:
        image (np.ndarray): Input BGR image.

    Returns:
        np.ndarray: Warped image after perspective transformation.
    """
    def _order_points(pts: np.ndarray) -> np.ndarray:
        """
        Orders the points of a contour into a consistent top-left, top-right, bottom-right, bottom-left format.

        Args:
            pts (np.ndarray): Array of points from a contour.

        Returns:
            np.ndarray: A 4x2 array of points in the order (top-left, top-right, bottom-right, bottom-left).
        """
        pts = pts.reshape(pts.shape[0], 2)
        rect = np.zeros((4, 2), dtype="float32")
        sum_coord = pts.sum(axis=1)
        rect[0] = pts[np.argmin(sum_coord)]
        rect[2] = pts[np.argmax(sum_coord)]
        diff_coord = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff_coord)]
        rect[3] = pts[np.argmax(diff_coord)]
        return rect
    
    max_contour = _find_max_contour(image)
    if max_contour is None:
        return image
    rect = top_left, top_right, bottom_right, bottom_left = _order_points(max_contour)
    width_a = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
    width_b = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))
    max_width = max(int(width_a), int(width_b))
    
    height_a = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))
    height_b = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))
    max_height = max(int(height_a), int(height_b))
    
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")
    
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    return warped

def _find_max_contour(image: np.ndarray) -> np.ndarray:
    """
    Finds the largest contour in an image by area.

    Args:
        image (np.ndarray): Input BGR image.

    Returns:
        np.ndarray: The largest contour found, or None if no contours are detected.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh_img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    return max(contours, key=cv2.contourArea) if contours else None