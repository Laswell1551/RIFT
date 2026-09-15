"""Attach observed multi-neighbor interaction sets to the frozen RIFT cache.

The script reads the already frozen windows and reconstructs, at the final
observed instant only, the K neighbors with the smallest constant-velocity CPA.
No future coordinate is used to select or encode a neighbor.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


LAT0, LON0 = 4.62, -74.10
MX = math.cos(math.radians(LAT0)) * 111_320.0
MY = 110_540.0
HORIZON = 10.0
FEATURE_DIM = 9


def to_xy(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    return np.column_stack(((lon - LON0) * MX, (lat - LAT0) * MY)).astype(np.float32)


def encode_queries(snapshot: pd.DataFrame, rows: np.ndarray, k_neighbors: int) -> np.ndarray:
    """Encode a permutation-ready neighbor set for cached query rows."""
    encoded = np.zeros((len(rows), k_neighbors, FEATURE_DIM), np.float32)
    if len(snapshot) < 2 or not len(rows):
        return encoded

    xy = to_xy(snapshot.lat_deg.to_numpy(), snapshot.lon_deg.to_numpy())
    xyz = np.column_stack(
        (xy[:, 0] / 50.0, xy[:, 1] / 50.0, snapshot.alt_m.to_numpy() / 15.0)
    ).astype(np.float32)
    track = np.deg2rad(snapshot.trk_deg.to_numpy())
    speed = snapshot.gs_ms.to_numpy()
    velocity = np.column_stack(
        (
            speed * np.sin(track) / 50.0,
            speed * np.cos(track) / 50.0,
            snapshot.vs_ms.to_numpy() / 15.0,
        )
    ).astype(np.float32)
    tree = cKDTree(xyz)
    query = np.column_stack((rows[:, 2] / 50.0, rows[:, 3] / 50.0, rows[:, 4] / 15.0))
    search_k = min(64, len(snapshot))
    _, neighbor_index = tree.query(query, k=search_k, workers=-1)
    neighbor_index = np.asarray(neighbor_index)
    if neighbor_index.ndim == 1:
        neighbor_index = neighbor_index[:, None]

    id_to_index = {
        int(drone_id): index for index, drone_id in enumerate(snapshot.drone_id.to_numpy())
    }
    self_index = np.asarray([id_to_index.get(int(row[0]), -1) for row in rows], np.int64)
    valid_query = self_index >= 0
    if not np.any(valid_query):
        return encoded

    safe_self = np.maximum(self_index, 0)
    relative_position = xyz[neighbor_index] - query[:, None, :]
    relative_velocity = velocity[neighbor_index] - velocity[safe_self][:, None, :]
    velocity_sq = np.sum(relative_velocity * relative_velocity, axis=2)
    tau = -np.sum(relative_position * relative_velocity, axis=2) / np.maximum(velocity_sq, 1e-12)
    tau = np.clip(tau, 0.0, HORIZON)
    cpa_position = relative_position + tau[:, :, None] * relative_velocity
    cpa_distance = np.linalg.norm(cpa_position, axis=2)
    self_mask = neighbor_index == safe_self[:, None]
    cpa_distance[self_mask | ~valid_query[:, None]] = np.inf

    for query_index, row in enumerate(rows):
        if not valid_query[query_index]:
            continue
        order = np.argsort(cpa_distance[query_index], kind="stable")
        order = order[np.isfinite(cpa_distance[query_index, order])][:k_neighbors]
        if not len(order):
            continue
        position = relative_position[query_index, order].copy()
        rel_velocity = relative_velocity[query_index, order].copy()
        angle = -float(row[5])
        cosine, sine = math.cos(angle), math.sin(angle)
        for block in (position, rel_velocity):
            horizontal = block[:, :2].copy()
            block[:, 0] = cosine * horizontal[:, 0] - sine * horizontal[:, 1]
            block[:, 1] = sine * horizontal[:, 0] + cosine * horizontal[:, 1]
        count = len(order)
        encoded[query_index, :count, :3] = position
        encoded[query_index, :count, 3:6] = rel_velocity
        encoded[query_index, :count, 6] = tau[query_index, order] / HORIZON
        encoded[query_index, :count, 7] = np.clip(
            cpa_distance[query_index, order], 0.0, 10.0
        ) / 10.0
        encoded[query_index, :count, 8] = 1.0
    return encoded


def enrich_task(
    data_root: Path, source_cache: Path, output_cache: Path, task: str, k_neighbors: int
) -> dict:
    split_names = ("train", "val", "test")
    archives = {}
    arrays = {}
    interactions = {}
    for split in split_names:
        archive = np.load(source_cache / f"{task}__{split}.npz")
        archives[split] = archive
        arrays[split] = {key: archive[key] for key in archive.files}
        interactions[split] = np.zeros(
            (len(archive["obs"]), k_neighbors, FEATURE_DIM), np.float32
        )

    files = sorted((data_root / "processed" / "states").glob(f"{task}__*.parquet"))
    if not files:
        raise FileNotFoundError(f"No state files found for {task}")
    columns = (
        "sim_time_s", "drone_id", "lat_deg", "lon_deg", "alt_m", "gs_ms", "vs_ms", "trk_deg"
    )
    for run_index, path in enumerate(files):
        frame = pd.read_parquet(path, columns=list(columns))
        frame_by_time = {int(time_s): block for time_s, block in frame.groupby("sim_time_s", sort=False)}
        for split in split_names:
            run_mask = arrays[split]["run_index"] == run_index
            selected = np.flatnonzero(run_mask)
            if not len(selected):
                continue
            meta = arrays[split]["meta"]
            for time_s in np.unique(meta[selected, 1].astype(np.int64)):
                indices = selected[meta[selected, 1].astype(np.int64) == time_s]
                snapshot = frame_by_time.get(int(time_s))
                if snapshot is None:
                    continue
                interactions[split][indices] = encode_queries(
                    snapshot, meta[indices], k_neighbors
                )
        print(f"{task}: encoded {path.name}", flush=True)

    output_cache.mkdir(parents=True, exist_ok=True)
    summary = {"task": task, "k_neighbors": k_neighbors, "splits": {}}
    for split in split_names:
        payload = dict(arrays[split])
        payload["interaction_set"] = interactions[split]
        np.savez_compressed(output_cache / f"{task}__{split}.npz", **payload)
        valid = interactions[split][..., -1].sum(axis=1)
        summary["splits"][split] = {
            "windows": int(len(valid)),
            "mean_valid_neighbors": float(valid.mean()),
            "fully_populated_fraction": float(np.mean(valid == k_neighbors)),
        }
        archives[split].close()
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=("baseline", "wind5", "speed15", "alt180", "speed10", "alt200"))
    parser.add_argument("--k-neighbors", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = {
        "protocol": "frozen RIFT windows plus observed-only top-K CPA neighbor sets",
        "source_cache": str(args.source_cache),
        "k_neighbors": args.k_neighbors,
        "feature_order": [
            "local_dx_50m", "local_dy_50m", "dz_15m", "local_dvx_50mps",
            "local_dvy_50mps", "dvz_15mps", "tau_cpa_over_10s",
            "cpa_distance_clipped_over_10", "valid_mask",
        ],
        "tasks": [],
    }
    for task in args.tasks:
        report["tasks"].append(
            enrich_task(args.data_root, args.source_cache, args.output_cache, task, args.k_neighbors)
        )
    args.output_cache.mkdir(parents=True, exist_ok=True)
    (args.output_cache / "manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
