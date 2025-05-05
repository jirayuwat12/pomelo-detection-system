from abc import ABC, abstractmethod

import cv2

from .base_detector_config import BaseDetectorConfig
from .detector_result import DetectorResult


class BaseDetector(ABC):
    """
    Base class for all detectors.
    This class defines the interface for all detectors.
    """

    def __init__(self, config: BaseDetectorConfig | None = None):
        """
        Initialize the detector with the given configuration.

        Args:
            config (DetectorConfig): Configuration for the detector.
        """
        self.config = config if config else BaseDetectorConfig()

    @abstractmethod
    def load_model(self):
        """
        Load the model.
        This method should be implemented by subclasses.
        """
        pass

    @abstractmethod
    def predict(self, image: cv2.Mat) -> DetectorResult:
        """
        Make a prediction on the given image.

        Args:
            image: The image to make a prediction on.

        Returns:
            The prediction result.
        """
        pass

    def predict_from_path(self, image_path: str) -> DetectorResult:
        """
        Make a prediction on the given image path.

        Args:
            image_path: The path to the image to make a prediction on.

        Returns:
            The prediction result.
        """
        image = cv2.imread(image_path)
        return self.predict(image)
