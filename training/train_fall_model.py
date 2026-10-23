"""Trains the team's own fall detection model with YOLO.

Usage (on Google Colab with GPU, or on a computer with an NVIDIA card):

    python training/train_fall_model.py --data datasets/falls/data.yaml --epochs 50

The dataset must be in YOLO format (images/ + labels/ + data.yaml), for
example exported from Roboflow Universe as "YOLOv11". At the end the best
model is copied to models/fall_model.pt; set FALL_MODEL=models/fall_model.pt
in .env to use it in the IoT module.
"""
import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from safevision_iot.vision.detectors import fall_class_ids  # noqa: E402


@dataclass
class TrainingConfig:
    data: Path
    base_model: str = "yolo11n.pt"   # pretrained on COCO: training starts from it (transfer learning)
    epochs: int = 50
    image_size: int = 640
    batch: int = 16
    output: Path = Path("models/fall_model.pt")
    device: str | None = None        # None = GPU if available, else CPU


class DatasetChecker:
    """Validates the dataset before spending hours training."""

    def __init__(self, data_yaml: Path):
        self.data_yaml = data_yaml
        self.spec = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))

    @property
    def class_names(self) -> dict[int, str]:
        names = self.spec.get("names", {})
        return dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}

    def split_folder(self, split: str) -> Path:
        """Folder of a split, resolved like Roboflow exports expect (relative to data.yaml)."""
        base = Path(self.spec.get("path") or ".")
        if not base.is_absolute():
            base = (self.data_yaml.parent / base).resolve()
        folder = base / self.spec[split]
        if not folder.exists():
            folder = (self.data_yaml.parent / self.spec[split]).resolve()
        return folder

    def check(self) -> list[str]:
        problems = []
        if not fall_class_ids(self.class_names):
            problems.append(f"No class means 'fall': {self.class_names}")
        for split in ("train", "val"):
            if split not in self.spec:
                problems.append(f"data.yaml has no '{split}' entry")
                continue
            folder = self.split_folder(split)
            images = [p for p in folder.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
            if not images:
                problems.append(f"No images found for '{split}' in {folder}")
        return problems

    def resolved_copy(self) -> Path:
        """data.yaml with absolute folders, so training works from any directory."""
        spec = {"train": str(self.split_folder("train")), "val": str(self.split_folder("val")),
                "names": self.class_names}
        if "test" in self.spec and self.split_folder("test").exists():
            spec["test"] = str(self.split_folder("test"))
        target = self.data_yaml.parent / "data.resolved.yaml"
        target.write_text(yaml.safe_dump(spec, allow_unicode=True), encoding="utf-8")
        return target


class FallModelTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def train(self) -> Path:
        from ultralytics import YOLO

        model = YOLO(self.config.base_model)
        results = model.train(data=str(self.config.data), epochs=self.config.epochs,
                              imgsz=self.config.image_size, batch=self.config.batch,
                              device=self.config.device, project="runs", name="fall_model",
                              exist_ok=True, plots=True)
        best = Path(results.save_dir) / "weights" / "best.pt"
        self.config.output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best, self.config.output)
        return self.config.output

    def evaluate(self, model_path: Path) -> dict[str, float]:
        from ultralytics import YOLO

        metrics = YOLO(str(model_path)).val(data=str(self.config.data), imgsz=self.config.image_size,
                                            device=self.config.device, plots=False, verbose=False)
        return {"precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
                "mAP50": float(metrics.box.map50), "mAP50-95": float(metrics.box.map)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the SafeVision AI fall model")
    parser.add_argument("--data", required=True, type=Path, help="path to the dataset data.yaml")
    parser.add_argument("--base-model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="0 for the first GPU, cpu for CPU")
    parser.add_argument("--output", type=Path, default=Path("models/fall_model.pt"))
    args = parser.parse_args()

    checker = DatasetChecker(args.data)
    problems = checker.check()
    if problems:
        sys.exit("Dataset problems:\n- " + "\n- ".join(problems))
    print(f"Classes: {checker.class_names} | fall classes: {fall_class_ids(checker.class_names)}")

    trainer = FallModelTrainer(TrainingConfig(checker.resolved_copy(), args.base_model, args.epochs, args.imgsz,
                                              args.batch, args.output, args.device))
    model_path = trainer.train()
    print(f"\nModel saved to {model_path}")
    for name, value in trainer.evaluate(model_path).items():
        print(f"  {name}: {value:.3f}")
    print(f"\nTo use it: FALL_MODEL={model_path} in .env")


if __name__ == "__main__":
    main()
