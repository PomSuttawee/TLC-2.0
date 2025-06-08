import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_image(image: np.ndarray, title: str = "Image", cmap: str = None) -> None:
    """
    Display an image using matplotlib.

    Args:
        image (np.ndarray): Image to display.
        title (str): Title of the image window.
        cmap (str): Colormap to use for displaying the image. Default is None.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
    plt.figure(figsize=(5, 5))
    plt.imshow(image_rgb, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()