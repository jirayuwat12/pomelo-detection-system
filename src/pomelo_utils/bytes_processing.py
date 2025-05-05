import cv2
import numpy as np


def bytes_to_cv2_image(byte_data: bytes) -> cv2.Mat:
    """
    Convert byte data to a CV2 image.

    Args:
        byte_data (bytes): The byte data to convert.

    Returns:
        numpy.ndarray: The converted CV2 image.
    """
    # Convert byte data to numpy array
    nparr = np.frombuffer(byte_data, np.uint8)

    # Decode the image from the numpy array
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    return img
