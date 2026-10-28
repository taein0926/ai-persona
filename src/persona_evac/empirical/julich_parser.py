from __future__ import annotations

import csv
import io
import json
import math
import statistics
import zipfile
from itertools import pairwise
from pathlib import Path

from persona_evac.hashing import sha256_file, sha256_value


FPS = 16.0
RESPONSE_THRESHOLD_MPS = 0.75
DOOR_MIDPOINTS_PX = [
    ((170.0 + 176.3) / 2, (379.0 + 283.5) / 2),
    ((163.8 + 166.4) / 2, (572.9 + 474.7) / 2),
    ((159.0 + 160.0) / 2, (775.1 + 670.7) / 2),
    ((1147.2 + 1151.2) / 2, (666.9 + 774.4) / 2),
    ((1142.4 + 1146.8) / 2, (469.0 + 570.3) / 2),
    ((1136.1 + 1138.4) / 2, (278.1 + 375.2) / 2),
]
METRES_PER_PIXEL = 10.0 / math.dist(DOOR_MIDPOINTS_PX[1], DOOR_MIDPOINTS_PX[4])


def parse_pair_file(path: Path) -> list[list[tuple[float, float]]]:
    """Parse Jülich alternating x/y rows into participant trajectories."""
    rows = [
        [float(value) for value in line.split()]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) % 2:
        raise ValueError(f"{path} has an odd number of coordinate rows")
    trajectories = []
    for index in range(0, len(rows), 2):
        if len(rows[index]) != len(rows[index + 1]):
            raise ValueError(f"{path} x/y row lengths differ")
        trajectories.append(list(zip(rows[index], rows[index + 1], strict=True)))
    return trajectories


def parse_experiment_csv(text: str) -> dict[str, list[tuple[float, float] | None]]:
    rows = list(csv.reader(io.StringIO(text)))
    if not rows or len(rows) < 3 or (len(rows) - 1) % 2:
        raise ValueError("Jülich file must contain a header and alternating x/y frame rows")
    names = rows[0]
    trajectories: dict[str, list[tuple[float, float] | None]] = {name: [] for name in names}
    for row_index in range(1, len(rows), 2):
        xs, ys = rows[row_index], rows[row_index + 1]
        if len(xs) != len(names) or len(ys) != len(names):
            raise ValueError("Coordinate row width differs from the header")
        for column, name in enumerate(names):
            try:
                point = (float(xs[column]), float(ys[column]))
            except ValueError:
                point = None
            trajectories[name].append(point)
    return trajectories


def _frame_speeds(
    trajectory: list[tuple[float, float] | None], metres_per_pixel: float = METRES_PER_PIXEL
) -> list[float | None]:
    speeds: list[float | None] = [None]
    for previous, current in pairwise(trajectory):
        speeds.append(
            math.dist(previous, current) * metres_per_pixel * FPS
            if previous is not None and current is not None
            else None
        )
    return speeds


def extract_experiment_features(
    text: str, *, group_id: str, treatment: str
) -> list[dict[str, object]]:
    trajectories = parse_experiment_csv(text)
    helper_crossings = []
    for helper in ("door1", "door2"):
        speeds = _frame_speeds(trajectories[helper])
        crossing = next(
            (index for index, speed in enumerate(speeds) if speed is not None and speed >= RESPONSE_THRESHOLD_MPS),
            None,
        )
        if crossing is not None:
            helper_crossings.append(crossing)
    if not helper_crossings:
        raise ValueError("Neither door helper crosses the response threshold")
    experiment_start = max(0, min(helper_crossings) - 8)
    records = []
    for participant in (str(value) for value in range(1, 13)):
        trajectory = trajectories[participant]
        speeds = _frame_speeds(trajectory)
        response_frame = next(
            (
                index
                for index, speed in enumerate(speeds)
                if index >= experiment_start
                and speed is not None
                and speed >= RESPONSE_THRESHOLD_MPS
            ),
            None,
        )
        if response_frame is None:
            raise ValueError(f"Participant {participant} never crosses the response threshold")
        valid_points = [(index, point) for index, point in enumerate(trajectory) if point is not None]
        final_point = valid_points[-1][1]
        exit_midpoint = min(DOOR_MIDPOINTS_PX, key=lambda point: math.dist(final_point, point))
        exit_frame = next(
            (
                index
                for index, point in enumerate(trajectory[response_frame:], start=response_frame)
                if point is not None
                and math.dist(point, exit_midpoint) * METRES_PER_PIXEL <= 2.0
            ),
            valid_points[-1][0],
        )
        movement_speeds = [
            speed
            for speed in speeds[response_frame : exit_frame + 1]
            if speed is not None and speed > 0
        ]
        if not movement_speeds:
            raise ValueError(f"Participant {participant} has no movement speed samples")
        records.append(
            {
                "record_id": f"{group_id}-{treatment}-{participant}",
                "treatment": treatment,
                "response_time_s": round((response_frame - experiment_start) / FPS, 6),
                "observed_speed_proxy_mps": round(statistics.median(movement_speeds), 6),
            }
        )
    return records


def prepare_julich_features(zip_path: Path, output: Path) -> Path:
    records: list[dict[str, object]] = []
    processed_files = []
    with zipfile.ZipFile(zip_path) as archive:
        names = sorted(
            name
            for name in archive.namelist()
            if name.endswith(("_individual.txt", "_group.txt"))
        )
        for name in names:
            stem = Path(name).stem
            group_id, treatment = stem.split("_", 1)
            text = archive.read(name).decode("utf-8-sig")
            records.extend(
                extract_experiment_features(text, group_id=group_id, treatment=treatment)
            )
            processed_files.append(name)
    if len(records) != 216:
        raise ValueError(f"Expected 216 participant records, got {len(records)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "record_id",
                "treatment",
                "response_time_s",
                "observed_speed_proxy_mps",
            ],
        )
        writer.writeheader()
        writer.writerows(records)
    output.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "source": "Bode et al., PLOS ONE 2015 supplementary data S1",
                "doi": "10.1371/journal.pone.0121227",
                "zip_sha256": sha256_file(zip_path),
                "processed_files": processed_files,
                "record_count": len(records),
                "metres_per_pixel": METRES_PER_PIXEL,
                "response_threshold_mps": RESPONSE_THRESHOLD_MPS,
                "feature_hash": sha256_value(records),
                "speed_notice": "Observed trajectory speed proxy; not free-flow desired-speed ground truth.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return output


def trajectory_features(
    points_px: list[tuple[float, float]],
    *,
    metres_per_pixel: float,
    experiment_start_frame: int,
    exit_midpoint_px: tuple[float, float],
    exit_zone_m: float = 2.0,
) -> tuple[float, float]:
    speeds = [
        math.dist(points_px[index - 1], points_px[index]) * metres_per_pixel * FPS
        for index in range(1, len(points_px))
    ]
    response_frame = next(
        (index for index, speed in enumerate(speeds, start=1) if speed >= RESPONSE_THRESHOLD_MPS),
        None,
    )
    if response_frame is None:
        raise ValueError("Participant never crosses the response speed threshold")
    response_time_s = max(0.0, (response_frame - experiment_start_frame) / FPS)
    end_frame = next(
        (
            index
            for index, point in enumerate(points_px[response_frame:], start=response_frame)
            if math.dist(point, exit_midpoint_px) * metres_per_pixel <= exit_zone_m
        ),
        len(points_px) - 1,
    )
    valid = sorted(speed for speed in speeds[response_frame - 1 : end_frame] if speed > 0)
    if not valid:
        raise ValueError("No valid movement speeds after response")
    middle = len(valid) // 2
    median = valid[middle] if len(valid) % 2 else (valid[middle - 1] + valid[middle]) / 2
    return response_time_s, median
