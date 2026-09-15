"""Descriptive paired road contrasts, retaining every method pair and metric."""
from itertools import combinations
import numpy as np
import pandas as pd
from analyze_stratification import ROOT, bootstrap


def main():
    frame = pd.read_csv(ROOT/'results/interaction_summary/per_seed.csv')
    methods = list(frame.method.drop_duplicates())
    metrics = ['final_fde', 'final_cpa_fde', 'ordinary_forgetting', 'cpa_forgetting', 'gap']
    rows = []
    for pi, (a, b) in enumerate(combinations(methods, 2)):
        va = frame[frame.method == a].set_index('seed').sort_index()
        vb = frame[frame.method == b].set_index('seed').sort_index()
        assert len(va) == len(vb) == 4 and va.index.equals(vb.index)
        for metric in metrics:
            difference = (va[metric] - vb[metric]).to_numpy()
            mean, lo, hi = bootstrap(difference, np.random.default_rng(900 + pi))
            rows.append({'method_a': a, 'method_b': b, 'contrast': 'a-minus-b',
                         'metric': metric, 'n_seeds': 4, 'mean': mean,
                         'ci_low': lo, 'ci_high': hi,
                         'interpretation': 'descriptive paired seed interval; no multiplicity adjustment'})
    pd.DataFrame(rows).to_csv(ROOT/'results/interaction_summary/all_pairwise_contrasts.csv', index=False)
    print('Saved all 30 descriptive paired contrasts.')


if __name__ == '__main__':
    main()
