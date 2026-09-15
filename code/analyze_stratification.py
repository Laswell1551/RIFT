"""Frozen-size and initial-difficulty controls for per-example forgetting."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'data/lowaltitude/cache'
TASKS = ['baseline', 'wind5', 'speed15', 'alt180', 'speed10', 'alt200']
METHODS = ['finetune', 'reservoir', 'gss_adapted', 'cgsm_dual', 'dualls_adapted']
SEEDS = [7, 11, 23, 47, 59, 71, 89, 107]


def cells_from_initial(run, initial):
    """Tie-preserving within-run quantile cells; no final errors used."""
    cells = np.empty(len(run), np.int16)
    for r in np.unique(run):
        idx = np.flatnonzero(run == r)
        cuts = np.unique(np.quantile(initial[idx], np.arange(1, 10) / 10))
        cells[idx] = int(r) * 10 + np.searchsorted(cuts, initial[idx], side='right')
    return cells


def control(delta, mask, cells, rng, draws=1000):
    count = int(mask.sum())
    expected = 0.
    sampled = np.zeros(draws)
    audit_indices = []
    audit_cells = []
    for c in np.unique(cells):
        pool = np.flatnonzero(cells == c)
        k = int(mask[pool].sum())
        audit_cells.append({'cell': int(c), 'pool_n': len(pool), 'selected_n': k})
        if not k:
            continue
        expected += k * float(delta[pool].mean())
        if k == len(pool):
            sampled += delta[pool].sum()
            audit_indices.extend(pool.tolist())
        else:
            for b in range(draws):
                take = rng.choice(pool, size=k, replace=False)
                sampled[b] += delta[take].sum()
                if b == 0:
                    audit_indices.extend(take.tolist())
    assert len(audit_indices) == count and len(set(audit_indices)) == count
    return expected / count, sampled / count, audit_indices, audit_cells


def bootstrap(values, rng, n=20000):
    v = np.asarray(values)
    b = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return float(v.mean()), *np.quantile(b, [.025, .975]).tolist()


def main():
    out = ROOT / 'results/stratification'
    out.mkdir(parents=True, exist_ok=True)
    rows, manifests, distributions = [], [], {}
    for mi, method in enumerate(METHODS):
        for seed in SEEDS:
            for ti, task in enumerate(TASKS[:-1]):
                source = ROOT / f'results/synthetic-seed{seed}/predictions'
                initial = np.load(source / f'{method}__seed{seed}__step{ti}__{task}.npz')['fde'].astype(float)
                final = np.load(source / f'{method}__seed{seed}__step5__{task}.npz')['fde'].astype(float)
                cache = np.load(CACHE / f'{task}__test.npz')
                mask = cache['risk_distance'] <= 1
                run = cache['run_index']
                assert len(initial) == len(mask) == 2000
                delta = final - initial
                matched = cells_from_initial(run, initial)
                row = {'method': method, 'seed': seed, 'task': task, 'n': len(mask), 'cpa_n': int(mask.sum()),
                       'overall': float(delta.mean()), 'cpa': float(delta[mask].mean()),
                       'initial_overall_fde': float(initial.mean()), 'initial_cpa_fde': float(initial[mask].mean())}
                for ci, (name, cells) in enumerate([('random', run), ('matched', matched)]):
                    random_seed = 2026091500 + mi * 100000 + seed * 100 + ti * 2 + ci
                    mean, draws, indices, counts = control(delta, mask, cells, np.random.default_rng(random_seed))
                    row[name] = mean
                    row[f'cpa_minus_{name}'] = row['cpa'] - mean
                    row[f'permutation_p_{name}'] = float((1 + np.sum(np.abs(draws - mean) >= abs(row['cpa'] - mean))) / (len(draws) + 1))
                    key = f'{method}__{seed}__{task}__{name}'
                    distributions[key] = draws
                    manifests.append({'key': key, 'rng_seed': random_seed, 'draws': 1000,
                                      'first_draw_indices': indices, 'cell_counts': counts,
                                      'cell_assignment': cells.tolist()})
                row['cpa_minus_overall'] = row['cpa'] - row['overall']
                rows.append(row)
        print(method, 'complete', flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'per_task.csv', index=False)
    metrics = ['overall', 'cpa', 'random', 'matched', 'cpa_minus_overall', 'cpa_minus_random', 'cpa_minus_matched']
    by_seed = frame.groupby(['method', 'seed'])[metrics].mean().reset_index()
    by_seed.to_csv(out / 'per_seed.csv', index=False)
    result = []
    for mi, method in enumerate(METHODS):
        part = by_seed[by_seed.method == method].sort_values('seed')
        for metric in metrics:
            mean, lo, hi = bootstrap(part[metric], np.random.default_rng(613 + mi))
            result.append({'method': method, 'metric': metric, 'n_seeds': len(part), 'mean': mean, 'ci_low': lo, 'ci_high': hi})
    summary = pd.DataFrame(result)
    summary.to_csv(out / 'summary.csv', index=False)
    np.savez_compressed(out / 'conditional_permutation_distributions.npz', **distributions)
    (out / 'control_audit.json').write_text(json.dumps(manifests), encoding='utf-8')
    print(summary[summary.metric == 'cpa_minus_matched'].to_string(index=False))


if __name__ == '__main__':
    main()
