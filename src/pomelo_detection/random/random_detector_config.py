from dataclasses import dataclass, field

from pomelo_detection.base_detector_config import BaseDetectorConfig


@dataclass
class RandomDetectorConfig(BaseDetectorConfig):
    """
    Configuration for the Pomelo Detector.
    """

    output_amount: int = 2
    class_names: list[str] = field(default_factory=lambda: ["young", "old", "pomelo"])
    bbox_range: list[tuple[int, int]] = field(default_factory=lambda: [(0, 0), (1000, 1000)])
    random_seed: int = 6969
