"""Preserve public release partitions; make fixed training subsets and input masks."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from analyze_stratification import ROOT


def row_hashes(trajectory, target):
    return [hashlib.sha256(np.ascontiguousarray(np.r_[t.ravel(), y], dtype=np.float32).tobytes()).hexdigest()
            for t, y in zip(trajectory, target)]


def main():
    cfg = json.loads((ROOT / 'configs/interaction.json').read_text())
    source = ROOT / 'data/interaction/interaction_merge'
    out = ROOT / 'data/interaction_cache'
    out.mkdir(parents=True,exist_ok=True)
    manifest, hashes = [], []
    for ti, task in enumerate(cfg['task_order']):
        split_hashes = {}
        for split in ['train', 'val']:
            path = source / split / f'{split}_{task}.npz'
            hashes.append({'file': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size,
                           'sha256': hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()})
            with np.load(path, allow_pickle=True) as a:
                n = len(a['nbagents'])
                keep = np.arange(n)
                if split == 'train' and n > cfg['train_limit_per_task']:
                    keep = np.sort(np.random.default_rng(cfg['training_subset_seed'] + ti).choice(n, cfg['train_limit_per_task'], replace=False))
                traj = a['trajectory'][keep].astype(np.float32)
                maps = a['maps'][keep].astype(np.float32)
                lane = np.stack([x.toarray() for x in a['lanefeature'][keep]]).astype(np.float32)
                nadj = np.stack([x.toarray() for x in a['adjacency'][keep]]).astype(np.float32)
                agents, splines = a['nbagents'][keep], a['nbsplines'][keep]
                target = a['intention'][keep].astype(np.float32)
            assert traj.shape[1:] == (26, 9, 8) and maps.shape[1:] == (55, 5, 5)
            assert np.all((agents >= 1) & (agents <= 26)) and np.all((splines >= 1) & (splines <= 55))
            valid = np.c_[np.arange(55)[None] < splines[:,None], np.arange(26)[None] < agents[:,None]]
            af = np.maximum(nadj, 0)
            af = af + af.transpose(0,2,1) + np.eye(55)[None] * valid[:,:55,None]
            af = np.linalg.matrix_power(af, 4) > 0
            relpos = traj[:,1:,-1,2:4] - traj[:,0:1,-1,2:4]
            relvel = traj[:,1:,-1,4:6] - traj[:,0:1,-1,4:6]
            tau = np.clip(-np.sum(relpos * relvel, axis=-1) / np.maximum(np.sum(relvel**2, axis=-1), 1e-12), 0, 3)
            distance = np.linalg.norm(relpos + tau[...,None] * relvel, axis=-1)
            distance[np.arange(1,26)[None] >= agents[:,None]] = np.inf
            cpa = distance.min(axis=1)
            # Input-only support: target history and observed nearest neighbors.
            current = np.linalg.norm(relpos, axis=-1)
            current[np.arange(1,26)[None] >= agents[:,None]] = np.inf
            order = np.argsort(current, axis=1, kind='stable')[:,:8]
            near = np.take_along_axis(traj[:,1:,-1,:6], order[:,:,None], axis=1)
            near[~np.isfinite(np.take_along_axis(current, order, axis=1))] = 0
            observable = np.c_[traj[:,0].reshape(len(keep),-1), near.reshape(len(keep),-1)]
            assert all(np.isfinite(x).all() for x in [traj,maps,lane,target,observable])
            split_hashes[split] = row_hashes(traj, target)
            np.savez_compressed(out / f'{task}__{split}.npz', trajectory=traj, maps=maps, lane=lane,
                af=af, valid=valid, target=target, cpa=cpa.astype(np.float32), observable=observable,
                source_index=keep, agents=agents, splines=splines)
            manifest.append({'task': task, 'split': split, 'source_n': n, 'used_n': len(keep),
                             'cpa_n': int((cpa < 4).sum()), 'no_neighbor_n': int((agents == 1).sum())})
            print(task, split, n, 'used', len(keep), 'CPA', int((cpa < 4).sum()), flush=True)
        overlaps = set(split_hashes['train']) & set(split_hashes['val'])
        manifest[-1]['exact_train_val_trajectory_target_overlap_hashes'] = len(overlaps)
    pd.DataFrame(manifest).to_csv(out / 'manifest.csv', index=False)
    pd.DataFrame(hashes).to_csv(out / 'source_sha256.csv', index=False)
    (out / 'config_snapshot.json').write_text(json.dumps(cfg, indent=2))


if __name__ == '__main__':
    main()
