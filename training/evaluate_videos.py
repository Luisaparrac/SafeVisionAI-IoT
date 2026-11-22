"""Evaluates a folder of recorded videos with the real pipeline, without using the backend.

Usage:
    python training/evaluate_videos.py grabaciones --output training/results/resultados_antes.csv
"""
import argparse
import csv
import logging
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from safevision_iot.vision.rules import (ABNORMAL_MOVEMENT, CHILD_ON_FURNITURE,  # noqa: E402
                                         CHILD_RISK_ZONE, COLLAPSE, FALL, PET_ON_FURNITURE)

HIT = "Acierto"
MISSED = "No detectado"
FALSE_ALARM = "Falsa alarma"
NO_EVENT = "Ninguno"


@dataclass(frozen=True)
class ClipLabel:
    """Expected event of a clip, deduced from its file name."""
    name: str
    expected: str | None

    PREFIXES = {
        "caida_al_frente_": FALL,
        "caida_al_lado_": FALL,
        "caida_atras_": FALL,
        "caida_silla_": FALL,
        "desmayo_silla_": FALL,
        "actividad_anormal_": ABNORMAL_MOVEMENT,
        "actividad_normal_": None,
        "mascota_mueble_": PET_ON_FURNITURE,
        "nino_mueble_": CHILD_ON_FURNITURE,
        "nino_zona_": CHILD_RISK_ZONE,
    }
    RISK_EVENTS = (FALL, COLLAPSE, ABNORMAL_MOVEMENT, PET_ON_FURNITURE, CHILD_ON_FURNITURE,
                   CHILD_RISK_ZONE)
    EQUIVALENT = {FALL: {FALL, COLLAPSE}}

    @classmethod
    def from_filename(cls, filename: str) -> "ClipLabel":
        name = Path(filename).name
        for prefix, expected in cls.PREFIXES.items():
            if name.lower().startswith(prefix):
                return cls(name, expected)
        raise ValueError(f"El video {name} no tiene un prefijo conocido")

    def outcome(self, detected: set[str]) -> str:
        """Immobility is only informative: it never counts as an error."""
        if self.expected is not None:
            accepted = self.EQUIVALENT.get(self.expected, {self.expected})
            return HIT if detected & accepted else MISSED
        return FALSE_ALARM if detected.intersection(self.RISK_EVENTS) else HIT


@dataclass(frozen=True)
class ClipResult:
    label: ClipLabel
    detected: tuple[str, ...]

    @property
    def outcome(self) -> str:
        return self.label.outcome(set(self.detected))


class VideoEvaluator:
    """Analyzes each video with a new pipeline, so no tracking state is shared."""

    def __init__(self, settings, evidence_dir: str):
        self._settings = replace(settings, evidence_dir=evidence_dir)

    def evaluate(self, path: Path) -> ClipResult:
        from safevision_iot.vision.factory import build_pipeline

        label = ClipLabel.from_filename(path.name)
        pipeline = build_pipeline(self._settings, client=None, zones=lambda: [])
        try:
            findings = pipeline.run_video(str(path), self._settings.analysis_fps)
        finally:
            pipeline.close()
        return ClipResult(label, tuple(sorted({f.event_type for f in findings})))


class EvaluationReport:
    """Builds the results table, the CSV file and the summary."""

    COLUMNS = ("clip", "evento_esperado", "eventos_detectados", "resultado")

    def __init__(self, results: list[ClipResult]):
        self._results = results

    def rows(self) -> list[tuple[str, str, str, str]]:
        return [(r.label.name, r.label.expected or NO_EVENT, ", ".join(r.detected) or NO_EVENT, r.outcome)
                for r in self._results]

    def count(self, outcome: str) -> int:
        return sum(r.outcome == outcome for r in self._results)

    @property
    def accuracy(self) -> float:
        return 100 * self.count(HIT) / len(self._results) if self._results else 0.0

    def save_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            writer.writerow(self.COLUMNS)
            writer.writerows(self.rows())

    def table(self) -> str:
        rows = [self.COLUMNS, *self.rows()]
        widths = [max(len(row[i]) for row in rows) for i in range(len(self.COLUMNS))]
        return "\n".join("  ".join(value.ljust(w) for value, w in zip(row, widths)) for row in rows)

    def summary(self) -> str:
        return (f"Total de clips: {len(self._results)}\n"
                f"Aciertos: {self.count(HIT)}\n"
                f"No detectados: {self.count(MISSED)}\n"
                f"Falsas alarmas: {self.count(FALSE_ALARM)}\n"
                f"Porcentaje de acierto: {self.accuracy:.1f} %")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evalúa videos grabados con el pipeline de detección")
    parser.add_argument("folder", type=Path, help="Carpeta con los videos .mp4")
    parser.add_argument("--output", type=Path, default=Path("training/results/resultados.csv"))
    args = parser.parse_args()

    videos = sorted(args.folder.glob("*.mp4"))
    if not videos:
        print(f"No se encontraron videos en {args.folder}")
        return 1

    from safevision_iot.settings import Settings

    logging.getLogger("safevision.events").setLevel(logging.ERROR)
    results: list[ClipResult] = []
    with tempfile.TemporaryDirectory() as evidence_dir:
        evaluator = VideoEvaluator(Settings.from_env(), evidence_dir)
        for number, video in enumerate(videos, 1):
            try:
                result = evaluator.evaluate(video)
            except ValueError as error:
                print(f"Omitido: {error}")
                continue
            results.append(result)
            print(f"[{number}/{len(videos)}] {video.name}: {result.outcome}", flush=True)

    report = EvaluationReport(results)
    report.save_csv(args.output)
    print(f"\n{report.table()}\n\n{report.summary()}\nResultados guardados en {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
