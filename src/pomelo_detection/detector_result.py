from collections import namedtuple
from dataclasses import dataclass, field

import cv2

coco_box_format = namedtuple(
    "coco_box_format",
    [
        "xmin",  # x coordinate of the top-left corner
        "ymin",  # y coordinate of the top-left corner
        "width",  # width of the bounding box
        "height",  # height of the bounding box
    ],
)


@dataclass
class DetectorResult:
    """
    Result of the Pomelo Detector.
    """

    image: cv2.Mat
    image_path: str | None = None
    boxes: list[coco_box_format] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    probabilities: list[float] = field(default_factory=list)

    def __post_init__(self):
        if len(self.boxes) != len(self.classes):
            raise ValueError("The number of boxes and classes must be the same.")

    def to_json(self) -> dict:
        """
        Convert the result to JSON format.

        Returns:
            dict: The result in JSON format.
        """
        return {
            "image_path": self.image_path,
            "results": [{"bbox": box, "class": cls} for box, cls in zip(self.boxes, self.classes)],
        }

    def add_result(self, box: coco_box_format, cls: str, prob: float = 0.0):
        """
        Add a result to the detector result.

        Args:
            box (coco_box_format): The bounding box in COCO format.
            cls (str): The class of the object.
            prob (float): The probability of the detection.
        """
        self.boxes.append(box)
        self.classes.append(cls)
        self.probabilities.append(prob)
