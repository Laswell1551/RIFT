"""Prepare leakage-resistant continual trajectory-prediction windows.

The source dataset remains read-only.  All outputs are written below
``--output-root``.  Traffic risk is computed at the final observed instant,
so no future position is used by the predictor or memory selector.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.spatial import cKDTree


TASKS = ("baseline", "wind5", "speed15", "alt180", "speed10", "alt200")
LAT0, LON0 = 4.62, -74.10
MX = math.cos(math.radians(LAT0)) * 111_320.0
MY = 110_540.0
OBS_STEPS, PRED_STEPS = 5, 10
SPLIT_NAMES = ("train", "val", "test")


def split_of_drone(drone_id: int) -> str:
    """Stable 70/10/20 group split shared by every task and run."""
    bucket = int(drone_id) % 10
    return "train" if bucket < 7 else ("val" if bucket == 7 else "test")


def to_xy(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    return np.column_stack(((lon - LON0) * MX, (lat - LAT0) * MY)).astype(np.float32)


def canonicalize(points_xy: np.ndarray, altitude: np.ndarray) -> tuple[np.ndarray, float]:
    """Canonicalize horizontal heading and express altitude relative to observation end."""
    origin = points_xy[OBS_STEPS - 1]
    heading = points_xy[OBS_STEPS - 1] - points_xy[0]
    angle = float(np.arctan2(heading[1], heading[0]))
    c, s = math.cos(-angle), math.sin(-angle)
    rotation = np.asarray(((c, -s), (s, c)), dtype=np.float32)
    horizontal = ((points_xy - origin) @ rotation.T).astype(np.float32)
    vertical = (altitude - altitude[OBS_STEPS - 1]).astype(np.float32)[:, None]
    return np.concatenate((horizontal, vertical), axis=1), angle


def extract_candidates(
    df: pd.DataFrame,
    split: str,
    max_drones: int,
    max_windows: int,
    seed: int,
    excluded_ids: set[int] | None = None,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    ids = np.asarray([d for d in df.drone_id.unique() if split_of_drone(int(d)) == split])
    if excluded_ids:
        ids = np.asarray([d for d in ids if int(d) not in excluded_ids])
    if len(ids) > max_drones:
        ids = rng.choice(ids, max_drones, replace=False)
    selected = df[df.drone_id.isin(ids)].sort_values(["drone_id", "sim_time_s"])

    obs, fut, context, meta = [], [], [], []
    length = OBS_STEPS + PRED_STEPS
    for drone_id, group in selected.groupby("drone_id", sort=False):
        times = group.sim_time_s.to_numpy()
        xy = to_xy(group.lat_deg.to_numpy(), group.lon_deg.to_numpy())
        altitude = group.alt_m.to_numpy(np.float32)
        groundspeed = group.gs_ms.to_numpy(np.float32)
        vertical_speed = group.vs_ms.to_numpy(np.float32)
        breaks = np.where(np.diff(times) != 1)[0] + 1
        for segment in np.split(np.arange(len(times)), breaks):
            if len(segment) < length:
                continue
            starts = np.arange(0, len(segment) - length + 1, PRED_STEPS)
            for start in starts:
                idx = segment[start : start + length]
                canonical, heading = canonicalize(xy[idx], altitude[idx])
                wind_speed = float(group.wind_speed_ms.iloc[idx[OBS_STEPS - 1]])
                wind_dir = math.radians(float(group.wind_dir_deg.iloc[idx[OBS_STEPS - 1]]))
                # Meteorological direction is converted to a Cartesian vector and
                # then rotated into the trajectory-canonical frame.
                wind_xy = np.asarray(
                    (-wind_speed * math.sin(wind_dir), -wind_speed * math.cos(wind_dir)),
                    dtype=np.float32,
                )
                c, s = math.cos(-heading), math.sin(-heading)
                wind_local = np.asarray(((c, -s), (s, c)), np.float32) @ wind_xy
                end = idx[OBS_STEPS - 1]
                obs.append(canonical[:OBS_STEPS])
                fut.append(canonical[OBS_STEPS:])
                context.append(
                    (
                        float(group.cruise_speed_ms.iloc[end]),
                        float(group.cruise_alt_ft.iloc[end]) * 0.3048,
                        float(wind_local[0]),
                        float(wind_local[1]),
                        float(groundspeed[end]),
                        float(vertical_speed[end]),
                    )
                )
                meta.append((
                    int(drone_id), int(times[end]), float(xy[end, 0]), float(xy[end, 1]),
                    float(altitude[end]), heading,
                ))

    if len(obs) > max_windows:
        keep = np.sort(rng.choice(len(obs), max_windows, replace=False))
    else:
        keep = np.arange(len(obs))
    return {
        "obs": np.asarray(obs, np.float32)[keep],
        "future": np.asarray(fut, np.float32)[keep],
        "context": np.asarray(context, np.float32)[keep],
        "meta": np.asarray(meta, np.float64)[keep],
    }


def observed_risk(
    df: pd.DataFrame, meta: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return current separation, constant-velocity CPA distance, and risk.

    Positions and velocities are normalized by 50 m horizontally and 15 m
    vertically. For each queried aircraft, closest point of approach (CPA) is
    computed over the next prediction horizon using a 64-entry nearest-aircraft
    query at the observation instant, followed by focal-aircraft exclusion,
    under constant velocity. A CPA distance below
    one enters the separation ellipsoid. The calculation uses no future label.
    """
    if not len(meta):
        empty = np.empty(0, np.float32)
        return empty, empty, empty, np.empty((0, 6), np.float32)
    by_time: dict[int, list[int]] = {}
    for i, row in enumerate(meta):
        by_time.setdefault(int(row[1]), []).append(i)
    current_nearest = np.full(len(meta), np.inf, np.float32)
    cpa_nearest = np.full(len(meta), np.inf, np.float32)
    interaction = np.zeros((len(meta), 6), np.float32)
    for time_s, candidate_idx in by_time.items():
        snapshot = df[df.sim_time_s == time_s]
        if len(snapshot) < 2:
            continue
        xy = to_xy(snapshot.lat_deg.to_numpy(), snapshot.lon_deg.to_numpy())
        xyz_scaled = np.column_stack((xy[:, 0] / 50.0, xy[:, 1] / 50.0, snapshot.alt_m.to_numpy() / 15.0))
        track = np.deg2rad(snapshot.trk_deg.to_numpy())
        speed = snapshot.gs_ms.to_numpy()
        velocity_scaled = np.column_stack((
            speed * np.sin(track) / 50.0,
            speed * np.cos(track) / 50.0,
            snapshot.vs_ms.to_numpy() / 15.0,
        )).astype(np.float32)
        tree = cKDTree(xyz_scaled)
        rows = meta[candidate_idx]
        query = np.column_stack((rows[:, 2] / 50.0, rows[:, 3] / 50.0, rows[:, 4] / 15.0))
        neighbor_count = min(64, len(snapshot))
        distances, neighbors = tree.query(query, k=neighbor_count, workers=-1)
        distances = np.atleast_2d(distances)
        neighbors = np.atleast_2d(neighbors)
        id_to_index = {int(drone_id): index for index, drone_id in enumerate(snapshot.drone_id.to_numpy())}
        self_index = np.asarray([id_to_index[int(row[0])] for row in rows], np.int64)
        self_mask = neighbors == self_index[:, None]
        relative_position = xyz_scaled[neighbors] - query[:, None, :]
        relative_velocity = velocity_scaled[neighbors] - velocity_scaled[self_index][:, None, :]
        velocity_sq = np.sum(relative_velocity * relative_velocity, axis=2)
        time_to_cpa = -np.sum(relative_position * relative_velocity, axis=2) / np.maximum(velocity_sq, 1e-12)
        time_to_cpa = np.clip(time_to_cpa, 0.0, float(PRED_STEPS))
        cpa_position = relative_position + time_to_cpa[:, :, None] * relative_velocity
        cpa_distance = np.linalg.norm(cpa_position, axis=2)
        cpa_distance[self_mask] = np.inf
        current_distance = distances.copy()
        current_distance[self_mask] = np.inf
        current_nearest[candidate_idx] = np.min(current_distance, axis=1).astype(np.float32)
        cpa_nearest[candidate_idx] = np.min(cpa_distance, axis=1).astype(np.float32)
        critical_local = np.argmin(cpa_distance, axis=1)
        row_index = np.arange(len(rows))
        critical_position = relative_position[row_index, critical_local].astype(np.float32)
        critical_velocity = relative_velocity[row_index, critical_local].astype(np.float32)
        angles = -rows[:, 5]
        cosine, sine = np.cos(angles), np.sin(angles)
        local_position = critical_position.copy()
        local_velocity = critical_velocity.copy()
        local_position[:, 0] = cosine * critical_position[:, 0] - sine * critical_position[:, 1]
        local_position[:, 1] = sine * critical_position[:, 0] + cosine * critical_position[:, 1]
        local_velocity[:, 0] = cosine * critical_velocity[:, 0] - sine * critical_velocity[:, 1]
        local_velocity[:, 1] = sine * critical_velocity[:, 0] + cosine * critical_velocity[:, 1]
        interaction[candidate_idx, :3] = local_position
        interaction[candidate_idx, 3:] = local_velocity
    risk = np.exp(-0.5 * cpa_nearest).astype(np.float32)
    risk[~np.isfinite(cpa_nearest)] = 0.0
    return current_nearest, cpa_nearest, risk, interaction


def build_task(
    state_root: Path,
    output_root: Path,
    task: str,
    max_drones: int,
    limits: dict[str, int],
    seed: int,
    excluded_test_ids: set[int],
) -> dict:
    files = sorted(state_root.glob(f"{task}__*.parquet"))
    if not files:
        raise FileNotFoundError(f"No state files found for task {task!r} below {state_root}")
    pooled = {split: {key: [] for key in ("obs", "future", "context", "interaction", "meta", "current_distance", "risk_distance", "risk_score", "run_index")} for split in SPLIT_NAMES}
    for run_index, path in enumerate(files):
        columns = [
            "cruise_speed_ms", "wind_dir_deg", "wind_speed_ms", "cruise_alt_ft",
            "sim_time_s", "drone_id", "lat_deg", "lon_deg", "alt_m", "gs_ms", "vs_ms", "trk_deg",
        ]
        df = pd.read_parquet(path, columns=columns)
        for split_index, split in enumerate(SPLIT_NAMES):
            candidates = extract_candidates(
                df, split, max_drones, limits[split], seed + 1000 * run_index + 17 * split_index,
                excluded_test_ids if split == "test" else None,
            )
            current_distance, cpa_distance, risk, interaction = observed_risk(df, candidates["meta"])
            candidates["interaction"] = interaction
            candidates["current_distance"] = current_distance
            candidates["risk_distance"] = cpa_distance
            candidates["risk_score"] = risk
            candidates["run_index"] = np.full(len(risk), run_index, np.int16)
            for key, values in candidates.items():
                pooled[split][key].append(values)

    output_root.mkdir(parents=True, exist_ok=True)
    summary = {"task": task, "files": [p.name for p in files], "splits": {}}
    for split in SPLIT_NAMES:
        merged = {key: np.concatenate(chunks, axis=0) for key, chunks in pooled[split].items()}
        rng = np.random.default_rng(seed + 10_000 + SPLIT_NAMES.index(split))
        if len(merged["obs"]) > limits[split]:
            keep = np.sort(rng.choice(len(merged["obs"]), limits[split], replace=False))
            merged = {key: values[keep] for key, values in merged.items()}
        target = output_root / f"{task}__{split}.npz"
        np.savez_compressed(target, **merged)
        finite = merged["risk_distance"][np.isfinite(merged["risk_distance"])]
        summary["splits"][split] = {
            "windows": int(len(merged["obs"])),
            "unique_drones": int(len(np.unique(merged["meta"][:, 0]))),
            "risk_distance_quantiles": np.quantile(finite, (0.1, 0.25, 0.5, 0.75, 0.9)).round(4).tolist() if len(finite) else [],
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True, help="Path to the regen dataset root")
    parser.add_argument("--output-root", type=Path, default=Path("data/cache/cgsm"))
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS)
    parser.add_argument("--max-drones-per-run", type=int, default=250)
    parser.add_argument("--max-train", type=int, default=4000)
    parser.add_argument("--max-val", type=int, default=1000)
    parser.add_argument("--max-test", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--exclude-test-ids", type=Path, help="JSON list of IDs used during development")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state_root = args.data_root / "processed" / "states"
    limits = {"train": args.max_train, "val": args.max_val, "test": args.max_test}
    excluded_test_ids = set()
    if args.exclude_test_ids:
        excluded_test_ids = {int(value) for value in json.loads(args.exclude_test_ids.read_text(encoding="utf-8"))}
    manifest = {
        "protocol": "drone-id grouped 70/10/20 split; 5 s 3-D observation; 10 s 3-D prediction",
        "seed": args.seed,
        "excluded_development_test_ids": len(excluded_test_ids),
        "tasks": [],
    }
    for task in args.tasks:
        print(f"Preparing {task} ...", flush=True)
        summary = build_task(
            state_root, args.output_root, task, args.max_drones_per_run, limits, args.seed, excluded_test_ids
        )
        manifest["tasks"].append(summary)
        print(json.dumps(summary["splits"], indent=2), flush=True)
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()

