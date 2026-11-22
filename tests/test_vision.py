"""Tests of the AI rules with synthetic detections (no camera or YOLO model needed)."""
from dataclasses import replace

import numpy as np

from safevision_iot.event_reporter import EventReporter
from safevision_iot.settings import Settings
from safevision_iot.vision.analyzer import Cooldown, SceneAnalyzer
from safevision_iot.vision.detections import (Box, Detection, FrameDetections, FurnitureDetection,
                                              PersonDetection)
from safevision_iot.vision.detectors import child_class_ids, crop_with_margin
from safevision_iot.vision.posture import Posture, PostureClassifier
from safevision_iot.vision.rules import (ABNORMAL_MOVEMENT, CHILD_ON_FURNITURE, CHILD_RISK_ZONE, FALL,
                                         IMMOBILITY, PET_ON_FURNITURE, RESTRICTED_ZONE,
                                         AbnormalMovementRule, ChildOnFurnitureRule, FallRule, Finding,
                                         ImmobilityRule, PetOnFurnitureRule, RestrictedZoneRule,
                                         TrainedFallRule)
from safevision_iot.vision.tracking import TrackedSubject, TrackRegistry
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


def upper_body_only(track=1) -> PersonDetection:
    """Person sitting very close to the camera: face, shoulders and arms only."""
    kps = np.zeros((17, 2))
    kps[0] = [500, 50]
    kps[5], kps[6] = [350, 300], [650, 300]
    kps[7], kps[8] = [250, 500], [750, 500]
    conf = np.where(kps.any(axis=1), 0.9, 0.0)
    return PersonDetection(track, "person", Box(0, 0, 900, 590), 0.9, kps, conf)


def test_partial_body_is_unknown_not_lying():
    assert PostureClassifier().classify(upper_body_only(), frame_height=H) == Posture.UNKNOWN


def test_lying_person_without_visible_legs_is_still_lying():
    person = lying()
    person.keypoint_conf[13:17] = 0.05            # the model loses the legs on the floor
    assert PostureClassifier().classify(person, frame_height=H) == Posture.LYING


def test_partial_body_never_triggers_a_fall():
    frames = [standing()] * 10 + [upper_body_only()] * 30
    assert FALL not in types(run(frames))


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


# ---------- abnormal movement ----------
def with_wrists(person: PersonDetection, wrist_y: float) -> PersonDetection:
    keypoints, conf = person.keypoints.copy(), person.keypoint_conf.copy()
    keypoints[9], keypoints[10] = [person.box.x1, wrist_y], [person.box.x2, wrist_y]
    conf[9], conf[10] = 0.9, 0.9
    return replace(person, keypoints=keypoints, keypoint_conf=conf)


def abnormal_scene():
    return analyzer(extra_rules=[AbnormalMovementRule(speed_threshold=2.5, min_peaks=3, window_seconds=3)])


def test_repeated_fast_arms_trigger_abnormal_movement():
    frames = [with_wrists(standing(), 150 if i % 2 else 400) for i in range(10)]
    frames += [with_wrists(standing(), 400)] * 15       # the movements are counted after settling
    findings = [f for f in run(frames, abnormal_scene()) if f.event_type == ABNORMAL_MOVEMENT]
    assert len(findings) == 1
    assert findings[0].confidence == 0.5
    assert "movimientos bruscos" in findings[0].description


def test_single_arm_movement_does_not_trigger():
    frames = [with_wrists(standing(), 400)] * 5 + [with_wrists(standing(), 150)] * 20
    assert ABNORMAL_MOVEMENT not in types(run(frames, abnormal_scene()))


def test_fast_arms_before_ending_up_lying_do_not_trigger():
    frames = [with_wrists(standing(), 150 if i % 2 else 400) for i in range(5)]
    frames += [with_wrists(lying(), 400)] * 20
    assert ABNORMAL_MOVEMENT not in types(run(frames, abnormal_scene()))


def test_fast_arms_while_hips_drop_do_not_trigger():
    frames = []
    for i in range(10):
        person = with_wrists(standing(), 150 if i % 2 else 400)
        person.keypoints[11:13, 1] = 360 if i % 2 else 260
        frames.append(person)
    frames += [with_wrists(standing(), 400)] * 15
    assert ABNORMAL_MOVEMENT not in types(run(frames, abnormal_scene()))


def test_lying_person_moving_arms_does_not_trigger():
    frames = [with_wrists(lying(), 300 if i % 2 else 500) for i in range(15)]
    assert ABNORMAL_MOVEMENT not in types(run(frames, abnormal_scene()))


def test_still_person_does_not_trigger_abnormal_movement():
    frames = [with_wrists(standing(), 300)] * 20
    assert ABNORMAL_MOVEMENT not in types(run(frames, abnormal_scene()))


# ---------- children and pets ----------
IMAGE = np.zeros((H, W, 3), dtype=np.uint8)
EVERYWHERE = Zone(9, "Cocina", ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))


def furniture(label, x1, y1, x2, y2) -> FurnitureDetection:
    return FurnitureDetection(None, label, Box(x1, y1, x2, y2), 0.8)


def sitting() -> PersonDetection:
    return replace(standing(), box=Box(200, 100, 400, 340))


def age_scene(rule, is_child=True):
    registry = TrackRegistry(PostureClassifier(), age_classifier=lambda image, box: (is_child, 0.87))
    return SceneAnalyzer(registry, [rule], Cooldown(60))


def run_furniture(frames, items, scene):
    found: list[Finding] = []
    for i, detection in enumerate(frames):
        found += scene.process(FrameDetections(i / FPS, W, H, [detection], items, IMAGE))
    return found


def pet_scene():
    return SceneAnalyzer(TrackRegistry(PostureClassifier()), [PetOnFurnitureRule(3)], Cooldown(60))


def test_age_needs_minimum_votes():
    subject = TrackedSubject(1, "person", min_age_votes=5)
    for _ in range(4):
        subject.add_age_vote(True, 0.9)
    assert not subject.is_child and subject.age_label is None
    subject.add_age_vote(True, 0.8)
    assert subject.is_child and subject.age_label == "niño (88 %)"


def test_age_uses_majority_of_the_last_15_votes():
    subject = TrackedSubject(1, "person")
    for _ in range(15):
        subject.add_age_vote(True, 0.9)
    for _ in range(8):
        subject.add_age_vote(False, 0.7)
    assert len(subject.age_votes) == 15
    assert not subject.is_child and subject.age_label == "adulto (70 %)"


def test_age_is_classified_at_most_twice_per_second():
    calls = []
    registry = TrackRegistry(PostureClassifier(),
                             age_classifier=lambda image, box: calls.append(1) or (True, 0.9))
    for i in range(FPS * 4):
        registry.update([standing()], i / FPS, W, H, IMAGE)
    assert len(calls) <= 8


def test_child_class_names():
    assert child_class_ids({0: "adults", 1: "children"}) == [1]
    assert child_class_ids({0: "adulto", 1: "nino"}) == [1]
    assert child_class_ids({0: "adult", 1: "not_child"}) == []


def test_crop_adds_margin_and_stays_inside_the_image():
    assert crop_with_margin(IMAGE, Box(100, 100, 200, 300)).shape[:2] == (240, 120)
    assert crop_with_margin(IMAGE, Box(0, 0, 1000, 600)).shape[:2] == (600, 1000)


def test_pet_on_couch_triggers():
    findings = run_furniture([dog()] * 20, [furniture("couch", 100, 420, 700, 600)], pet_scene())
    assert types(findings) == [PET_ON_FURNITURE]
    assert findings[0].detected_class == "dog" and "perro sobre sofá" in findings[0].description


def test_pet_on_the_floor_next_to_couch_does_not_trigger():
    assert run_furniture([dog()] * 30, [furniture("couch", 300, 420, 900, 600)], pet_scene()) == []


def test_pet_on_the_floor_in_front_of_couch_does_not_trigger():
    assert run_furniture([dog()] * 30, [furniture("couch", 100, 300, 700, 520)], pet_scene()) == []


def test_pet_on_a_chair_does_not_trigger():
    assert run_furniture([dog()] * 30, [furniture("chair", 100, 420, 700, 600)], pet_scene()) == []


def test_child_on_bed_triggers_as_person():
    findings = run_furniture([standing()] * 40, [furniture("bed", 100, 300, 700, 560)],
                             age_scene(ChildOnFurnitureRule(2)))
    assert types(findings) == [CHILD_ON_FURNITURE]
    assert findings[0].detected_class == "person" and "niño (87 %)" in findings[0].description
    assert findings[0].is_child


def test_adult_on_bed_does_not_trigger():
    assert run_furniture([standing()] * 40, [furniture("bed", 100, 300, 700, 560)],
                         age_scene(ChildOnFurnitureRule(2), is_child=False)) == []


def test_child_sitting_on_chair_does_not_trigger():
    assert run_furniture([sitting()] * 40, [furniture("chair", 150, 200, 450, 500)],
                         age_scene(ChildOnFurnitureRule(2))) == []


def test_child_standing_on_chair_triggers():
    findings = run_furniture([standing()] * 40, [furniture("chair", 150, 200, 450, 500)],
                             age_scene(ChildOnFurnitureRule(2)))
    assert types(findings) == [CHILD_ON_FURNITURE]


def test_child_on_the_floor_does_not_trigger():
    assert run_furniture([standing()] * 40, [furniture("bed", 500, 300, 900, 560)],
                         age_scene(ChildOnFurnitureRule(2))) == []


def test_child_in_zone_waits_for_votes_and_reports_child_event():
    findings = run_furniture([standing()] * 30, [], age_scene(RestrictedZoneRule(lambda: [EVERYWHERE], 2)))
    assert types(findings) == [CHILD_RISK_ZONE]
    assert findings[0].detected_class == "person" and "niño (87 %)" in findings[0].description


def test_adult_in_zone_keeps_the_normal_event():
    findings = run_furniture([standing()] * 30, [],
                             age_scene(RestrictedZoneRule(lambda: [EVERYWHERE], 2), is_child=False))
    assert types(findings) == [RESTRICTED_ZONE]
    assert "adulto (87 %)" in findings[0].description


def test_zone_event_is_not_lost_if_age_never_arrives():
    registry = TrackRegistry(PostureClassifier(), age_classifier=lambda image, box: None)
    scene = SceneAnalyzer(registry, [RestrictedZoneRule(lambda: [EVERYWHERE], 2)], Cooldown(60))
    assert types(run_furniture([standing()] * 40, [], scene)) == [RESTRICTED_ZONE]


def test_reporter_uses_child_subject_id():
    reporter = EventReporter(None, 1, "unused", person_subject_id=7, child_subject_id=11)
    child = reporter.to_event(Finding(CHILD_ON_FURNITURE, 1, "person", 0.9, "x", is_child=True))
    adult = reporter.to_event(Finding(FALL, 2, "person", 0.9, "y"))
    assert (child.subject_id, child.detected_class) == (11, "person")
    assert adult.subject_id == 7


def test_reporter_builds_evidence_url(tmp_path):
    from pathlib import Path
    reporter = EventReporter(None, 1, str(tmp_path), evidence_base_url="http://localhost:8080/evidence/")
    event = reporter.to_event(Finding(FALL, 1, "person", 0.9, "x"), Path("a_Caída.jpg"))
    assert event.evidence_url == "http://localhost:8080/evidence/a_Ca%C3%ADda.jpg"


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