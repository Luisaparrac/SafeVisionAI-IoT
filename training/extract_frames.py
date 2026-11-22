"""Extracts N photos per second from each video, using OpenCV.

Usage:
    python training/extract_frames.py grabaciones frames --fps 1
"""
import argparse
from pathlib import Path

import cv2


class FrameExtractor:
    """Saves photos named <video_name>_<number>.jpg in the output folder."""

    DEFAULT_VIDEO_FPS = 25.0

    def __init__(self, output_dir: Path, fps: float):
        if fps <= 0:
            raise ValueError("Las fotos por segundo deben ser mayores que 0")
        self._output_dir = output_dir
        self._fps = fps

    def extract_folder(self, folder: Path) -> int:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        total = 0
        for video in sorted(folder.glob("*.mp4")):
            saved = self.extract(video)
            print(f"{video.name}: {saved} fotos", flush=True)
            total += saved
        return total

    def extract(self, video: Path) -> int:
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            print(f"No se pudo abrir {video.name}")
            return 0
        video_fps = capture.get(cv2.CAP_PROP_FPS) or self.DEFAULT_VIDEO_FPS
        period = video_fps / self._fps
        index, saved = 0, 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index >= saved * period:
                saved += 1
                cv2.imwrite(str(self._output_dir / f"{video.stem}_{saved:04d}.jpg"), frame)
            index += 1
        capture.release()
        return saved


def main() -> int:
    parser = argparse.ArgumentParser(description="Extrae fotos de los videos para el dataset")
    parser.add_argument("videos", type=Path, help="Carpeta con los videos .mp4")
    parser.add_argument("output", type=Path, help="Carpeta donde se guardan las fotos")
    parser.add_argument("--fps", type=float, default=1.0, help="Fotos por segundo de video")
    args = parser.parse_args()

    total = FrameExtractor(args.output, args.fps).extract_folder(args.videos)
    print(f"Se extrajeron {total} fotos en {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
