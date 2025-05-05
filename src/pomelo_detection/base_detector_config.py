from dataclasses import dataclass


@dataclass
class BaseDetectorConfig:
    """
    Configuration for the Pomelo Detector.
    """

    # Path to the model weights
    device: str = "cpu"
