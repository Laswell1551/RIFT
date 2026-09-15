"""Recompute paper summaries from the released metric tables, without raw data."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from analyze_stratification import bootstrap

ROOT=Path(__file__).resolve().parents[1]

def main():
    d=ROOT/'results/interaction_summary'
    source=pd.read_csv(d/'all_checkpoint_metrics.csv')
    expected=pd.read_csv(d/'per_seed.csv').set_index(['method','seed']).sort_index()
    assert len(source)==576
    rows=[]
    for (method,seed),f in source.groupby(['method','seed']):
        assert len(f)==36 and all((f.learn_step==i).sum()==i+1 for i in range(8))
        assert not f.duplicated(['learn_step','eval_task_index']).any()
        last=f[f.learn_step==7].sort_values('eval_task_index')
        first=f[f.learn_step==f.eval_task_index].sort_values('eval_task_index')
        diff=last[['fde','risk_fde']].to_numpy()-first[['fde','risk_fde']].to_numpy()
        rows.append(dict(method=method,seed=seed,final_fde=last.fde.mean(),final_cpa_fde=last.risk_fde.mean(),
            ordinary_forgetting=diff[:-1,0].mean(),cpa_forgetting=diff[:-1,1].mean(),gap=(diff[:-1,1]-diff[:-1,0]).mean()))
    actual=pd.DataFrame(rows).set_index(['method','seed']).sort_index()
    np.testing.assert_allclose(actual[expected.columns],expected,atol=1e-12,rtol=1e-12)
    # The published aggregate intervals are seed bootstraps, not window bootstraps.
    summary=pd.read_csv(d/'summary.csv')
    methods=list(pd.read_csv(d/'per_seed.csv').method.drop_duplicates())
    for i,method in enumerate(methods):
        for metric in actual.columns:
            values=actual.loc[method,metric].to_numpy()
            mean,lo,hi=bootstrap(values,np.random.default_rng(456+i))
            ref=summary[(summary.method==method)&(summary.metric==metric)].iloc[0]
            np.testing.assert_allclose([mean,lo,hi],[ref['mean'],ref.ci_low,ref.ci_high],atol=1e-12,rtol=1e-12)
    for folder,seeds in [('stratification',8),('application',8),('application_incident',8)]:
        f=pd.read_csv(ROOT/f'results/{folder}/per_seed.csv')
        assert all(f.groupby('method').seed.nunique()==seeds)
        assert not f.duplicated(['method','seed']).any()
    print(json.dumps({'checkpoint_rows':576,'trained_runs':16,'road_summaries_and_intervals':'PASS',
                      'synthetic_and_detection_seed_completeness':'PASS'},indent=2))

if __name__=='__main__':main()
