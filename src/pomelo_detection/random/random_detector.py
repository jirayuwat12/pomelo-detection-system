import random

import cv2

from pomelo_detection.base_detector import BaseDetector
from pomelo_detection.detector_result import DetectorResult

from .random_detector_config import RandomDetectorConfig


class RandomDetector(BaseDetector):
    """
    Random Detector for Pomelo.
    This detector randomly generates bounding boxes and classes.
    """

    def __init__(self, config: RandomDetectorConfig | None = None):
        """
        Initialize the detector with the given configuration.
        """
        super().__init__()
        self.config = config if config else RandomDetectorConfig()
        # Set the random seed for reproducibility
        random.seed(self.config.random_seed)

    def load_model(self):
        """
        This detector does not require a model to be loaded.
        """
        pass

    def predict(self, image: cv2.Mat) -> DetectorResult:
        """
        Make a prediction on the given image.
        This method randomly generates bounding boxes and classes.
        """
        result = DetectorResult(image=image)

        # Get the image dimensions
        height, width, _ = image.shape

        # Generate random bounding boxes and classes
        for _ in range(self.config.output_amount):
            # Randomly select coordinates for the bounding box
            x1 = random.randint(self.config.bbox_range[0][0], min(self.config.bbox_range[1][0], width))
            y1 = random.randint(self.config.bbox_range[0][1], min(self.config.bbox_range[1][1], height))
            x2 = random.randint(x1, min(self.config.bbox_range[1][0], width))
            y2 = random.randint(y1, min(self.config.bbox_range[1][1], height))
            # Randomly select a class ID
            class_id = random.choice(range(len(self.config.class_names)))

            # Append the bounding box and class ID to the result
            result.add_result(
                box=(x1, y1, x2 - x1, y2 - y1),
                cls=class_id,
            )

        return result
