"""Click-to-label tool for shinai keypoints, writing YOLO pose labels."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..config import Config
from ..constants import ENTER_KEY, ESCAPE_KEY
from ..viz.constants import FONT
from .constants import (GRAB_RADIUS, IMAGE_EXTENSIONS, MAX_VIEW_SIZE, POINT_COLORS, POINTS_PER_SHINAI, SHINAI_HELP,
                        SHINAI_WINDOW_TITLE, VALUES_PER_LINE)


def label_path(image_path: Path) -> Path:
    """Ultralytics layout: .../images/x.jpg is labelled by .../labels/x.txt."""
    parts = list(image_path.parts)
    if "images" not in parts:
        return image_path.with_suffix(".txt")
    last_images_folder = len(parts) - 1 - parts[::-1].index("images")
    parts[last_images_folder] = "labels"
    return Path(*parts).with_suffix(".txt")


def load_labels(path: Path, width: int, height: int) -> list[np.ndarray]:
    """Each shinai as a (3, 2) pixel array: butt, tsuba, tip."""
    shinais = []
    for line in path.read_text().splitlines():
        values = line.split()
        if len(values) >= VALUES_PER_LINE:
            # skip the class and box, then drop each point's visibility flag
            points = np.array(values[5:VALUES_PER_LINE], float).reshape(POINTS_PER_SHINAI, 3)[:, :2]
            shinais.append(points * (width, height))
    return shinais


def save_labels(path: Path, shinais: list[np.ndarray], width: int, height: int) -> None:
    """One line per shinai: class, box, then x y visibility per keypoint, all normalised to 0..1."""
    lines = []
    for points in shinais:
        points = np.clip(points, 0, (width - 1, height - 1))
        # the box is the points' extent plus a margin, since a shinai seen end-on has almost no extent
        margin = 0.02 * max(width, height)
        left, top = np.maximum(points.min(axis=0) - margin, 0)
        right, bottom = np.minimum(points.max(axis=0) + margin, (width, height))
        box = ((left + right) / 2 / width, (top + bottom) / 2 / height, (right - left) / width, (bottom - top) / height)
        fields = [f"{value:.6f}" for value in box]
        fields += [f"{point_x / width:.6f} {point_y / height:.6f} 2" for point_x, point_y in points]  # 2 = visible
        lines.append("0 " + " ".join(fields))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


class _Prefill:
    """Suggests shinai from pose and the wrist baseline, or from the trained model once it exists."""

    def __init__(self, config: Config):
        from ..perception.pose import PoseEstimator
        from ..perception.shinai import make_shinai_estimator

        from ..perception.player_filter import PlayerFilter

        self.pose_estimator = PoseEstimator(config.model)
        self.player_filter = PlayerFilter(config.track) if config.track.filter_players else None
        self.shinai_estimator = make_shinai_estimator(config)

    def __call__(self, image: np.ndarray) -> list[np.ndarray]:
        from ..perception.tracker import untracked_players

        detections = self.pose_estimator(image)
        if self.player_filter is not None:
            detections = self.player_filter(image, detections)  # so shinpan are never suggested
        players = untracked_players(detections)
        self.shinai_estimator.reset()  # images are unrelated, so no history carries over
        self.shinai_estimator(image, players)
        return [np.stack([player.shinai.butt, player.shinai.tsuba, player.shinai.tip])
                for player in players if player.shinai is not None]


def run(images: str, config: Config, prefill: bool = True) -> None:
    image_paths = sorted(path for path in Path(images).rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        raise SystemExit(f"No images under {images}")
    suggest = _Prefill(config) if prefill else None
    shinais: list[np.ndarray] = []  # finished shinai on the current image
    pending: list[np.ndarray] = []  # points clicked so far for the next shinai
    view = {"zoom": 1.0, "dragging": None}  # dragging: (shinai index, point index) being moved

    def on_mouse(event, mouse_x, mouse_y, flags, userdata):
        cursor = np.array([mouse_x, mouse_y], float) / view["zoom"]  # screen pixels -> image pixels
        # find the existing point nearest the cursor, and whether it is close enough to grab
        point_ids = [(shinai_index, point_index) for shinai_index in range(len(shinais))
                     for point_index in range(POINTS_PER_SHINAI)]

        def distance_to_cursor(point_id: tuple[int, int]) -> float:
            shinai_index, point_index = point_id
            return float(np.linalg.norm(shinais[shinai_index][point_index] - cursor))

        nearest = min(point_ids, key=distance_to_cursor, default=None)
        # the grab radius is in screen pixels, so convert the image-pixel distance back
        on_a_point = nearest is not None and distance_to_cursor(nearest) * view["zoom"] < GRAB_RADIUS

        if event == cv2.EVENT_LBUTTONDOWN:
            if on_a_point:
                view["dragging"] = nearest
            else:
                pending.append(cursor)
                if len(pending) == POINTS_PER_SHINAI:  # third click completes a shinai
                    shinais.append(np.array(pending))
                    pending.clear()
        elif event == cv2.EVENT_MOUSEMOVE and view["dragging"] is not None:
            shinai_index, point_index = view["dragging"]
            shinais[shinai_index][point_index] = cursor
        elif event == cv2.EVENT_LBUTTONUP:
            view["dragging"] = None
        elif event == cv2.EVENT_RBUTTONDOWN and on_a_point:
            del shinais[nearest[0]]

    def render(image: np.ndarray, title: str) -> np.ndarray:
        zoom = view["zoom"]
        canvas = cv2.resize(image, None, fx=zoom, fy=zoom) if zoom != 1.0 else image.copy()
        for points in shinais:
            on_screen = (points * zoom).astype(int)
            cv2.line(canvas, tuple(on_screen[0]), tuple(on_screen[2]), (0, 220, 255), 1, cv2.LINE_AA)
            for point_index in range(POINTS_PER_SHINAI):
                cv2.circle(canvas, tuple(on_screen[point_index]), 5, POINT_COLORS[point_index], -1, cv2.LINE_AA)
        for point_index, point in enumerate(pending):  # hollow circles for a shinai still being clicked
            cv2.circle(canvas, tuple((point * zoom).astype(int)), 5, POINT_COLORS[point_index], 1, cv2.LINE_AA)
        for line_index, text in enumerate((title, SHINAI_HELP)):
            position = (8, 20 + 20 * line_index)
            cv2.putText(canvas, text, position, FONT, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, text, position, FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

    cv2.namedWindow(SHINAI_WINDOW_TITLE)
    cv2.setMouseCallback(SHINAI_WINDOW_TITLE, on_mouse)
    # resume at the first image that has no label file yet
    image_index = next((index for index, path in enumerate(image_paths) if not label_path(path).exists()),
                       len(image_paths) - 1)
    while 0 <= image_index < len(image_paths):
        image_path = image_paths[image_index]
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        labels_file = label_path(image_path)
        already_labelled = labels_file.exists()
        if already_labelled:
            shinais[:] = load_labels(labels_file, width, height)
        else:
            shinais[:] = suggest(image) if suggest else []
        pending.clear()
        view["zoom"] = min(1.0, MAX_VIEW_SIZE[0] / width, MAX_VIEW_SIZE[1] / height)
        view["dragging"] = None
        title = f"{image_index + 1}/{len(image_paths)}  {image_path.name}"
        if not already_labelled:
            title += "  (suggested, unsaved)"

        step = None  # +1 or -1 once the user moves on
        while step is None:
            cv2.imshow(SHINAI_WINDOW_TITLE, render(image, title))
            key = cv2.waitKey(20) & 0xFF
            if key in (ord(" "), ENTER_KEY):
                save_labels(labels_file, shinais, width, height)
                step = 1
            elif key == ord("b"):
                step = -1
            elif key == ord("x"):  # skip without saving
                step = 1
            elif key == ord("u") and pending:
                pending.pop()
            elif key == ord("c"):
                shinais.clear()
                pending.clear()
            elif key in (ord("q"), ESCAPE_KEY):
                cv2.destroyAllWindows()
                return
        image_index += step
    cv2.destroyAllWindows()
    print("Reached the end of the image list.")
