"""Trains the child/adult image classifier with YOLO11n-cls.

Usage:
    python training/train_child_model.py --data <folder> --epochs 30 --device 0

The folder must have one subfolder per class inside train/ and val/ (or valid/),
as Roboflow exports it with version.download("folder").
"""
import argparse
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass
class ChildTrainingConfig:
    data: Path
    base_model: str = "yolo11n-cls.pt"
    epochs: int = 30
    image_size: int = 224
    batch: int = 32
    output: Path = Path("models/child_model.pt")
    device: str | None = None


class ClassificationDatasetChecker:
    """Leaves the dataset with train/ and val/ folders, ready for YOLO classification."""

    def __init__(self, root: Path, val_ratio: float = 0.2, seed: int = 42):
        self.root = root
        self._val_ratio = val_ratio
        self._seed = seed

    @property
    def train_dir(self) -> Path:
        return self.root / "train"

    @property
    def val_dir(self) -> Path:
        return self.root / "val"

    def prepare(self) -> int:
        """Renames valid to val, or creates val from train. Returns the images moved."""
        if not self.train_dir.is_dir():
            raise FileNotFoundError(f"No existe la carpeta train/ en {self.root}")
        valid = self.root / "valid"
        if not self.val_dir.exists() and valid.is_dir():
            valid.rename(self.val_dir)
        if self._count_images(self.val_dir):
            return 0
        return self._create_validation()

    def check(self) -> list[str]:
        problems = []
        train_classes = self._classes(self.train_dir)
        if len(train_classes) < 2:
            problems.append(f"train/ necesita al menos 2 clases: {sorted(train_classes)}")
        missing = train_classes - self._classes(self.val_dir)
        if missing:
            problems.append(f"Faltan clases en val/: {sorted(missing)}")
        return problems

    def _create_validation(self) -> int:
        generator = random.Random(self._seed)
        moved = 0
        for class_dir in sorted(p for p in self.train_dir.iterdir() if p.is_dir()):
            images = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
            if len(images) < 2:
                continue
            target = self.val_dir / class_dir.name
            target.mkdir(parents=True, exist_ok=True)
            for image in generator.sample(images, max(1, int(len(images) * self._val_ratio))):
                shutil.move(str(image), str(target / image.name))
                moved += 1
        return moved

    @staticmethod
    def _classes(folder: Path) -> set[str]:
        if not folder.is_dir():
            return set()
        return {p.name for p in folder.iterdir()
                if p.is_dir() and any(f.suffix.lower() in IMAGE_SUFFIXES for f in p.iterdir())}

    @staticmethod
    def _count_images(folder: Path) -> int:
        if not folder.is_dir():
            return 0
        return sum(p.suffix.lower() in IMAGE_SUFFIXES for p in folder.rglob("*"))


class ChildModelTrainer:
    """Trains the classifier and copies the best weights to the models folder."""

    def __init__(self, config: ChildTrainingConfig):
        self.config = config
        self.save_dir: Path | None = None

    def train(self) -> Path:
        from ultralytics import YOLO

        model = YOLO(self.config.base_model)
        results = model.train(data=str(self.config.data.resolve()), epochs=self.config.epochs,
                              imgsz=self.config.image_size, batch=self.config.batch,
                              device=self.config.device, project="runs", name="child_model",
                              exist_ok=True, plots=True)
        self.save_dir = Path(results.save_dir)
        self.config.output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(self.save_dir / "weights" / "best.pt", self.config.output)
        return self.config.output

    def evaluate(self, model_path: Path) -> float:
        from ultralytics import YOLO

        metrics = YOLO(str(model_path)).val(data=str(self.config.data.resolve()),
                                            imgsz=self.config.image_size, device=self.config.device,
                                            plots=False, verbose=False)
        return float(metrics.top1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena el clasificador niño/adulto")
    parser.add_argument("--data", required=True, type=Path, help="carpeta del dataset")
    parser.add_argument("--base-model", default="yolo11n-cls.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default=None, help="0 para la primera GPU, cpu para el procesador")
    parser.add_argument("--output", type=Path, default=Path("models/child_model.pt"))
    args = parser.parse_args()

    checker = ClassificationDatasetChecker(args.data)
    try:
        moved = checker.prepare()
    except FileNotFoundError as error:
        sys.exit(str(error))
    if moved:
        print(f"No había validación: se movieron {moved} imágenes de train a val")
    problems = checker.check()
    if problems:
        sys.exit("Problemas en el dataset:\n- " + "\n- ".join(problems))

    trainer = ChildModelTrainer(ChildTrainingConfig(args.data, args.base_model, args.epochs, args.imgsz,
                                                    args.batch, args.output, args.device))
    model_path = trainer.train()
    print(f"\nModelo guardado en {model_path}")
    print(f"  top1_acc: {trainer.evaluate(model_path):.3f}")
    print(f"Resultados del entrenamiento en: {trainer.save_dir}")
    print(f"Para usarlo: CHILD_MODEL={model_path} en .env")


if __name__ == "__main__":
    main()
