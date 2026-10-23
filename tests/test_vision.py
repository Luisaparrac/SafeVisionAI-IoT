"""Tests of the AI rules with synthetic detections (no camera or YOLO model needed)."""
from dataclasses import replace

import numpy as np

from safevision_iot.event_reporter import EventReporter
from safevision_iot.settings import Settings
from safevision_iot.vision.analyzer import Cooldown, SceneAnalyzer
from safevision_iot.vision.detections import Box, Detection, FrameDetections, PersonDetection
from safevision_iot.vision.posture import Posture, PostureClassifier
from safevision_iot.vision.rules import (FALL, IMMOBILITY, RESTRICTED_ZONE, FallRule, Finding,
                                         ImmobilityRule, RestrictedZoneRule, TrainedFallRule)
from safevision_iot.vision.tracking import TrackRegistry
from safevision_iot.vision.zone import Zone
from safevision_iot.vision.zone_provider import ZoneProvider

W, H, FPS = 1000, 600, 5


def standing(x=300.0, track=1) -> PersonDetection:
    kps = np.zeros((17, 2))
    kps[5], kps[6] = [x - 20, 150], [x + 20, 150]      # shoulders
    kps[11], kps[12] = [x - 15, 260], [x + 15, 260]    # hips
    kps[0] = [x, 110]
    kps[15], kps[16] = [x - 15, 400], [x + 15, 400]    # ankles
    return PersonDetection(track, "person", Box(x - 50, 100, x + 50, 420), 0.9,
                           kps, np.where(kps.any(axis=1), 0.9, 0.0))


def lying(x=300.0, track=1) -> PersonDetection:
    kps = np.zeros((17, 2))
    kps[5], kps[6] = [x - 120, 390], [x - 120, 410]
    kps[11], kps[12] = [x, 390], [x, 410]
    kps[0] = [x - 160, 400]
    kps[15], kps[16] = [x + 130, 390], [x + 130, 410]
    return PersonDetection(track, "person", Box(x - 170, 370, x + 150, 430), 0.9,
                           kps, np.where(kps.any(axis=1), 0.9, 0.0))


def dog(x=200.0, track=100_001) -> Detection:
    return Detection(track, "dog", Box(x - 40, 480, x + 40, 560), 0.85)


def analyzer(zones=(), person_still=20, pet_still=60, extra_rules=()):
    rules = [FallRule(), ImmobilityRule(person_still, pet_still),
             RestrictedZoneRule(lambda: list(zones), dwell=2), *extra_rules]
    return SceneAnalyzer(TrackRegistry(PostureClassifier()), rules, Cooldown(60))


def run(frames, scene=None, fall_boxes=None):
    scene = scene or analyzer()
    found: list[Finding] = []
    for i, items in enumerate(frames):
        items = items if isinstance(items, list) else [items]
        found += scene.process(FrameDetections(i / FPS, W, H, items), fall_boxes)
    return found


def types(findings):
    return [f.event_type for f in findings]


# ---------- posture ----------
def test_posture():
    classifier = PostureClassifier()
    assert classifier.classify(standing()) == Posture.STANDING
    assert classifier.classify(lying()) == Posture.LYING


# ---------- fall ----------
def test_fall_is_detected_once():
    walk = [standing(300 + i * 4) for i in range(10)]
    assert types(run(walk + [lying()] * 25)).count(FALL) == 1


def test_gradual_fall_with_intermediate_frames():
    half_down = replace(standing(), box=Box(250, 300, 350, 560))
    half_down.keypoints[11:13, 1] = 470                    # hips already lower, still "standing"
    frames = [standing()] * 10 + [half_down] * 2 + [lying()] * 20
    assert types(run(frames)).count(FALL) == 1


def test_quick_recovery_is_not_a_fall():
    assert FALL not in types(run([standing()] * 10 + [lying()] * 3 + [standing()] * 5))


def test_lying_from_the_start_is_not_a_fall():
    assert FALL not in types(run([lying()] * 30))


def test_fall_detected_when_tracker_changes_id():
    walk = [standing(300 + i * 4, track=1) for i in range(10)]
    fallen = [replace(lying(), track_id=None)] * 3 + [lying(track=7)] * 22
    assert types(run(walk + fallen)).count(FALL) == 1


# ---------- immobility ----------
def test_person_immobility_after_limit():
    findings = run([standing()] * (FPS * 25))          # 25 s still, limit 20 s
    assert types(findings) == [IMMOBILITY]
    assert findings[0].detected_class == "person"


def test_pet_has_longer_immobility_limit():
    assert IMMOBILITY not in types(run([dog()] * (FPS * 25)))     # limit 60 s
    assert IMMOBILITY in types(run([dog()] * (FPS * 65)))


def test_walking_person_generates_nothing():
    assert run([standing(100 + (i % 40) * 8) for i in range(FPS * 30)]) == []


# ---------- restricted zone ----------
STAIRS = Zone(5, "Escalera", ((0.0, 0.7), (0.4, 0.7), (0.4, 1.0), (0.0, 1.0)))


def test_zone_contains():
    assert STAIRS.contains(0.2, 0.9)
    assert not STAIRS.contains(0.6, 0.9)


def test_pet_in_restricted_zone():
    scene = analyzer(zones=[STAIRS])
    findings = run([dog(x=200)] * 15, scene)            # paws at x=0.2, y=0.93
    assert types(findings) == [RESTRICTED_ZONE]
    assert findings[0].zone_id == 5 and findings[0].detected_class == "dog"


def test_brief_pass_through_zone_is_ignored():
    frames = [dog(x=200)] * 5 + [dog(x=800)] * 10       # 1 s inside, then out
    assert RESTRICTED_ZONE not in types(run(frames, analyzer(zones=[STAIRS])))


def test_zone_from_api_row():
    zone = Zone.from_api({"zone_id": 1, "name": "Cocina", "coordinates": [[0, 0], [1, 0], [1, 1]]})
    assert zone.zone_id == 1 and len(zone.polygon) == 3


def test_zone_provider_keeps_active_zones_of_its_camera():
    class Client:
        def list_zones(self):
            return [{"zone_id": 1, "name": "A", "coordinates": [[0, 0], [1, 0], [1, 1]], "camera_id": 1, "is_active": True},
                    {"zone_id": 2, "name": "B", "coordinates": [[0, 0], [1, 0], [1, 1]], "camera_id": 2, "is_active": True},
                    {"zone_id": 3, "name": "C", "coordinates": [[0, 0], [1, 0], [1, 1]], "camera_id": 1, "is_active": False}]
    assert [z.zone_id for z in ZoneProvider(Client(), camera_id=1)()] == [1]


# ---------- trained model ----------
def test_trained_fall_rule_confirms_after_seconds():
    scene = SceneAnalyzer(TrackRegistry(PostureClassifier()), [TrainedFallRule(confirm=2)], Cooldown(60))
    fall_box = [(Box(130, 370, 450, 430), 0.88)]
    findings = run([standing()] * 5 + [lying()] * 15, scene, fall_boxes=fall_box)
    assert types(findings) == [FALL]
    assert findings[0].confidence == 0.88


def test_fall_classes_of_public_datasets():
    from safevision_iot.vision.detectors import fall_class_ids
    assert fall_class_ids({0: "Fall-Detected", 1: "Walking", 2: "Sitting"}) == [0]
    assert fall_class_ids({0: "fallen", 1: "no_fall", 2: "person"}) == [0]
    assert fall_class_ids({0: "person"}) == []


# ---------- cooldown and reporter ----------
def test_cooldown_blocks_repeats():
    cooldown = Cooldown(60)
    assert cooldown.allow(1, FALL, 0)
    assert not cooldown.allow(1, FALL, 30)
    assert cooldown.allow(1, FALL, 61)


def test_reporter_uses_subject_ids_by_class():
    reporter = EventReporter(None, camera_id=3, evidence_dir="unused", person_subject_id=7, pet_subject_id=9)
    person = reporter.to_event(Finding(FALL, 1, "person", 0.912345, "x"))
    pet = reporter.to_event(Finding(RESTRICTED_ZONE, 2, "dog", 0.8, "y", zone_id=4))
    assert (person.subject_id, person.confidence, person.camera_id) == (7, 0.912, 3)
    assert (pet.subject_id, pet.zone_id, pet.detected_class) == (9, 4, "dog")


def test_settings_defaults(monkeypatch):
    monkeypatch.setenv("FALL_MODEL", "")
    monkeypatch.setenv("PERSON_SUBJECT_ID", "")
    settings = Settings.from_env()
    assert settings.fall_model is None and settings.person_subject_id is None
