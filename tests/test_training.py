"""Tests of the video evaluator without real videos or YOLO models."""
import csv

import pytest

from safevision_iot.vision.rules import ABNORMAL_MOVEMENT, FALL, IMMOBILITY
from training.evaluate_videos import (FALSE_ALARM, HIT, MISSED, ClipLabel, ClipResult,
                                      EvaluationReport)
from training.train_fall_model import DatasetChecker


@pytest.mark.parametrize("name", ["caida_al_frente_1.mp4", "caida_al_lado_2.mp4", "caida_atras_3.mp4",
                                  "caida_silla_1.mp4", "desmayo_silla_1.mp4"])
def test_fall_prefixes(name):
    assert ClipLabel.from_filename(name).expected == FALL


def test_abnormal_prefix():
    assert ClipLabel.from_filename("actividad_anormal_4.mp4").expected == ABNORMAL_MOVEMENT


def test_normal_prefix_expects_nothing():
    assert ClipLabel.from_filename("actividad_normal_1.mp4").expected is None


def test_unknown_prefix_is_rejected():
    with pytest.raises(ValueError):
        ClipLabel.from_filename("otro_video.mp4")


def test_expected_event_found_is_a_hit():
    assert ClipLabel("c.mp4", FALL).outcome({FALL}) == HIT


def test_expected_event_missing():
    assert ClipLabel("c.mp4", FALL).outcome(set()) == MISSED
    assert ClipLabel("c.mp4", FALL).outcome({ABNORMAL_MOVEMENT}) == MISSED


@pytest.mark.parametrize("event", [FALL, ABNORMAL_MOVEMENT])
def test_risk_event_on_normal_clip_is_a_false_alarm(event):
    assert ClipLabel("n.mp4", None).outcome({event}) == FALSE_ALARM


def test_immobility_is_not_an_error():
    assert ClipLabel("n.mp4", None).outcome({IMMOBILITY}) == HIT


def test_report_summary_and_csv(tmp_path):
    report = EvaluationReport([
        ClipResult(ClipLabel("a.mp4", FALL), (FALL,)),
        ClipResult(ClipLabel("b.mp4", FALL), ()),
        ClipResult(ClipLabel("c.mp4", None), (ABNORMAL_MOVEMENT,)),
        ClipResult(ClipLabel("d.mp4", None), (IMMOBILITY,)),
    ])
    assert (report.count(HIT), report.count(MISSED), report.count(FALSE_ALARM)) == (2, 1, 1)
    assert report.accuracy == 50

    path = tmp_path / "out" / "r.csv"
    report.save_csv(path)
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.reader(file))
    assert rows[0] == ["clip", "evento_esperado", "eventos_detectados", "resultado"]
    assert rows[2] == ["b.mp4", FALL, "Ninguno", MISSED]


def test_checker_finds_images_of_roboflow_export(tmp_path):
    dataset = tmp_path / "export" / "dataset"
    for split in ("train", "valid"):
        (dataset / split / "images").mkdir(parents=True)
        (dataset / split / "images" / "photo.jpg").write_bytes(b"fake")
    data_yaml = dataset / "data.yaml"
    data_yaml.write_text("train: ../train/images\nval: ../valid/images\nnames: [fall, person]\n",
                         encoding="utf-8")

    checker = DatasetChecker(data_yaml)
    assert checker.check() == []
    assert checker.split_folder("train") == (dataset / "train" / "images").resolve()
    assert checker.split_folder("val") == (dataset / "valid" / "images").resolve()
