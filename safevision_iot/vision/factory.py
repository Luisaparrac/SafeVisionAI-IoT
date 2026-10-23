"""Builds the whole AI pipeline from the settings (composition root)."""
from safevision_iot.backend_client import BackendClient
from safevision_iot.event_reporter import EventReporter
from safevision_iot.settings import Settings
from safevision_iot.vision.analyzer import Cooldown, SceneAnalyzer
from safevision_iot.vision.detectors import AnimalDetector, PoseDetector, TrainedFallDetector
from safevision_iot.vision.pipeline import DetectionPipeline, Preview
from safevision_iot.vision.posture import PostureClassifier
from safevision_iot.vision.rules import FallRule, ImmobilityRule, RestrictedZoneRule, TrainedFallRule
from safevision_iot.vision.tracking import TrackRegistry
from safevision_iot.vision.zone_provider import ZoneProvider


def build_pipeline(settings: Settings, client: BackendClient | None, show: bool = False,
                   zones=None) -> DetectionPipeline:
    """client=None runs the AI without sending anything (dry run)."""
    if zones is None:
        zones = (ZoneProvider(client, settings.camera_id, settings.zones_refresh_seconds)
                 if client else (lambda: []))

    rules = [
        FallRule(settings.fall_window_seconds, settings.fall_drop_ratio, settings.fall_confirm_seconds),
        ImmobilityRule(settings.person_immobility_seconds, settings.pet_immobility_seconds),
        RestrictedZoneRule(zones, settings.zone_dwell_seconds),
    ]
    fall_detector = None
    if settings.fall_model:
        fall_detector = TrainedFallDetector(settings.fall_model)
        rules.append(TrainedFallRule(settings.fall_confirm_seconds))

    analyzer = SceneAnalyzer(TrackRegistry(PostureClassifier()), rules,
                             Cooldown(settings.event_cooldown_seconds))
    reporter = EventReporter(client, settings.camera_id, settings.evidence_dir,
                             settings.person_subject_id, settings.pet_subject_id)
    detectors = [PoseDetector(settings.pose_model), AnimalDetector(settings.animal_model)]
    return DetectionPipeline(detectors, analyzer, reporter, fall_detector,
                             Preview() if show else None)
