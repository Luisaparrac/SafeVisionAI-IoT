"""Body posture from the keypoints and the box shape."""
from enum import Enum

import numpy as np

from safevision_iot.vision.detections import (LEFT_HIP, LEFT_SHOULDER, RIGHT_HIP, RIGHT_SHOULDER,
                                              PersonDetection)


class Posture(Enum):
    STANDING = "standing"
    SITTING = "sitting"
    LYING = "lying"
    UNKNOWN = "unknown"      

    @property
    def label(self) -> str:
        return {"standing": "de pie", "sitting": "sentado", "lying": "acostado",
                "unknown": "cuerpo no visible"}[self.value]


class PostureClassifier:
    """Decides if a person is standing, sitting or lying.

    It uses the torso angle (shoulders to hips) when those keypoints are
    visible, and the height/width ratio of the box as a fallback.
    """

    LYING_ANGLE = 60      # degrees from vertical
    STANDING_ANGLE = 35

    def classify(self, person: PersonDetection, frame_height: int | None = None) -> Posture:
        if not self.is_body_visible(person, frame_height):
            # Only the face or the torso is in the image (the person is very close
            # to the camera, or behind a desk): guessing would cause false falls.
            return Posture.UNKNOWN
        ratio = person.box.height / person.box.width
        angle = self.torso_angle(person)
        if angle is not None:
            if angle > self.LYING_ANGLE or ratio < 0.8:
                return Posture.LYING
            if angle < self.STANDING_ANGLE and ratio > 1.4:
                return Posture.STANDING
            return Posture.SITTING
        if ratio < 0.8:
            return Posture.LYING
        return Posture.STANDING if ratio > 1.6 else Posture.SITTING

    @staticmethod
    def is_body_visible(person: PersonDetection, frame_height: int | None = None) -> bool:
        """Is enough of the body in the image to judge the posture?

        No when the hips are not visible, or when the person fills the image from
        top to bottom (sitting right in front of the camera). Legs are not
        required: when someone lies on the floor the model often loses them.
        """
        if person.keypoints is None:
            return True     # no keypoints at all: fall back to the box shape
        if not (person.has_keypoints(LEFT_HIP) or person.has_keypoints(RIGHT_HIP)):
            return False
        if frame_height:
            touches_top = person.box.y1 <= 0.02 * frame_height
            touches_bottom = person.box.y2 >= 0.98 * frame_height
            if (touches_top and touches_bottom) or person.box.height >= 0.9 * frame_height:
                return False
        return True

    @staticmethod
    def torso_angle(person: PersonDetection) -> float | None:
        """0 = upright torso, 90 = horizontal torso."""
        if not person.has_keypoints(LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP):
            return None
        kps = person.keypoints
        shoulders = (kps[LEFT_SHOULDER] + kps[RIGHT_SHOULDER]) / 2
        hips = (kps[LEFT_HIP] + kps[RIGHT_HIP]) / 2
        dx, dy = shoulders - hips
        return float(np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6)))

    @staticmethod
    def hip_height(person: PersonDetection) -> float:
        """Vertical position of the hips in pixels (grows downwards)."""
        if person.has_keypoints(LEFT_HIP, RIGHT_HIP):
            return float((person.keypoints[LEFT_HIP][1] + person.keypoints[RIGHT_HIP][1]) / 2)
        return person.box.center[1]