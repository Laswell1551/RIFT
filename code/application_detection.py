"""Observed-neighbor eligibility, actual-future labels, fixed CV-neighbor alarms."""
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from analyze_stratification import ROOT, CACHE, TASKS, METHODS, SEEDS, bootstrap

SCALE = np.array([50., 50., 15.])
MX, MY = math.cos(math.radians(4.62)) * 111320., 110540.


def prepare():
    out = ROOT / 'data/application'
    out.mkdir(parents=True, exist_ok=True)
    report = []
    for task in TASKS[:-1]:
        a = np.load(CACHE / f'{task}__test.npz')
        b = np.load(ROOT / f'data/lowaltitude/metadata_cache/{task}__test.npz')
        assert np.array_equal(a['meta'], b['meta'][:, :5])
        assert np.array_equal(a['future'], b['future'])
        meta, n = b['meta'], len(a['obs'])
        cv = np.full((n, 8, 10, 3), np.nan, np.float32)
        event = np.zeros(n, bool)
        ongoing = np.zeros(n, bool)
        eligible = np.zeros(n, bool)
        ids_out = np.full((n, 8), -1, np.int64)
        gt_min = np.full(n, np.nan)
        raw_error = 0.
        files = sorted((ROOT / 'data/lowaltitude/states').glob(f'{task}__*.parquet'))
        assert len(files) == 5
        for ri, p in enumerate(files):
            f = pd.read_parquet(p, columns=['sim_time_s', 'drone_id', 'lat_deg', 'lon_deg', 'alt_m', 'gs_ms', 'vs_ms', 'trk_deg'])
            xyz = np.column_stack(((f.lon_deg.to_numpy() - (-74.10)) * MX, (f.lat_deg.to_numpy() - 4.62) * MY, f.alt_m.to_numpy()))
            # Match the float32 horizontal conversion used by the frozen cache.
            xyz[:, :2] = xyz[:, :2].astype(np.float32)
            f[['x','y','z']] = xyz
            indexed = f.set_index(['sim_time_s', 'drone_id'])[['x','y','z']]
            assert indexed.index.is_unique
            groups = f.groupby('sim_time_s', sort=False).indices
            selected = np.flatnonzero(a['run_index'] == ri)
            for t in np.unique(meta[selected, 1].astype(int)):
                qi = selected[meta[selected, 1].astype(int) == t]
                snap = f.iloc[groups[t]].sort_values('drone_id')
                pos = snap[['x','y','z']].to_numpy()
                ids = snap.drone_id.to_numpy(dtype=np.int64)
                trk = np.deg2rad(snap.trk_deg.to_numpy())
                vel = np.column_stack((snap.gs_ms.to_numpy() * np.sin(trk), snap.gs_ms.to_numpy() * np.cos(trk), snap.vs_ms.to_numpy()))
                tree = cKDTree(pos / SCALE)
                for ix in qi:
                    origin = meta[ix, 2:5]
                    distance_cut, _ = tree.query(origin / SCALE, k=min(10, len(pos)))
                    # Include all cutoff ties before applying the declared ID tie-break.
                    near = np.asarray(tree.query_ball_point(origin / SCALE, np.max(distance_cut) + 1e-10))
                    near = near[ids[near] != int(meta[ix, 0])]
                    dist = np.linalg.norm((pos[near] - origin) / SCALE, axis=1)
                    near = near[np.lexsort((ids[near], dist))][:8]
                    if len(near) != 8:
                        continue
                    ids_out[ix] = ids[near]
                    times = np.arange(t + 1, t + 11)
                    idx = pd.MultiIndex.from_arrays([np.tile(times, 9), np.repeat(np.r_[int(meta[ix,0]), ids[near]], 10)])
                    paths = indexed.reindex(idx).to_numpy().reshape(9, 10, 3)
                    if not np.isfinite(paths).all():
                        continue
                    angle = float(meta[ix, 5])
                    c, s = math.cos(-angle), math.sin(-angle)
                    rotation = np.array([[c, -s], [s, c]])
                    local_gt = paths[0] - origin
                    local_gt[:, :2] = local_gt[:, :2] @ rotation.T
                    raw_error = max(raw_error, float(np.abs(local_gt - a['future'][ix]).max()))
                    assert np.allclose(local_gt, a['future'][ix], atol=.002), (task, ix, raw_error)
                    future_dist = np.linalg.norm((paths[1:] - paths[0][None]) / SCALE, axis=2)
                    gt_min[ix] = future_dist.min()
                    event[ix] = gt_min[ix] <= 1.
                    ongoing[ix] = np.any(np.linalg.norm((pos[near] - origin) / SCALE, axis=1) <= 1.)
                    local_cv = pos[near, None, :] + vel[near, None, :] * np.arange(1, 11)[None, :, None] - origin
                    local_cv[..., :2] = local_cv[..., :2] @ rotation.T
                    cv[ix] = local_cv
                    eligible[ix] = True
            print('prepared', p.name, flush=True)
        np.savez_compressed(out / f'{task}.npz', neighbor_cv=cv, event=event, ongoing=ongoing,
                            eligible=eligible, neighbor_ids=ids_out, min_actual_distance=gt_min)
        report.append({'task': task, 'windows': n, 'eligible': int(eligible.sum()), 'excluded_incomplete': int((~eligible).sum()),
                       'events': int((event & eligible).sum()), 'nonevents': int((~event & eligible).sum()),
                       'ongoing_at_observation': int((ongoing & eligible).sum()), 'max_reconstructed_target_error_m': raw_error})
    pd.DataFrame(report).to_csv(out / 'label_support.csv', index=False)
    print(pd.DataFrame(report).to_string(index=False))


def rates(pred, data):
    eligible, event = data['eligible'], data['event']
    alarm = np.linalg.norm((pred[:, None] - data['neighbor_cv']) / SCALE, axis=-1).min(axis=(1, 2)) <= 1.
    pos, neg = eligible & event, eligible & ~event
    tp, fn = int((alarm & pos).sum()), int((~alarm & pos).sum())
    fp, tn = int((alarm & neg).sum()), int((~alarm & neg).sum())
    return {'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
            'fnr': fn / (tp + fn) if tp + fn else np.nan, 'fpr': fp / (fp + tn) if fp + tn else np.nan}


def analyze(incident=False):
    out = ROOT / ('results/application_incident' if incident else 'results/application')
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for method in METHODS:
        for seed in SEEDS:
            for ti, task in enumerate(TASKS[:-1]):
                archive = np.load(ROOT / f'data/application/{task}.npz')
                data = {k: archive[k] for k in archive.files}
                if incident:
                    data['eligible'] = data['eligible'] & ~data['ongoing']
                row = {'method': method, 'seed': seed, 'task': task}
                for name, step in [('initial', ti), ('final', 5)]:
                    pred = np.load(ROOT / f'results/synthetic-seed{seed}/predictions/{method}__seed{seed}__step{step}__{task}.npz')['prediction']
                    row.update({f'{name}_{k}': v for k, v in rates(pred, data).items()})
                for metric in ['fnr', 'fpr']:
                    row[f'delta_{metric}'] = row[f'final_{metric}'] - row[f'initial_{metric}']
                rows.append(row)
    f = pd.DataFrame(rows)
    f.to_csv(out / 'per_task.csv', index=False)
    metrics = ['initial_fnr', 'final_fnr', 'delta_fnr', 'initial_fpr', 'final_fpr', 'delta_fpr']
    s = f.groupby(['method', 'seed'])[metrics].mean().reset_index()
    s.to_csv(out / 'per_seed.csv', index=False)
    rows = []
    for mi, method in enumerate(METHODS):
        for metric in metrics:
            mean, lo, hi = bootstrap(s[s.method == method][metric], np.random.default_rng(314 + mi))
            rows.append({'method': method, 'metric': metric, 'mean': mean, 'ci_low': lo, 'ci_high': hi})
    pd.DataFrame(rows).to_csv(out / 'summary.csv', index=False)
    print(pd.DataFrame(rows).query("metric == 'delta_fnr'").to_string(index=False))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'analyze', 'incident'])
    args = parser.parse_args()
    prepare() if args.stage == 'prepare' else analyze(incident=args.stage == 'incident')
