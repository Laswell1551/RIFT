"""Post-primary CPA-threshold sensitivity on frozen historical predictions.

Only evaluation membership changes. Predictors, memories, task order, initial
error deciles and simulation-run identities remain fixed. Equal task weights;
eight training seeds are the bootstrap units. No new training is performed.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from analyze_stratification import ROOT,CACHE,TASKS,METHODS,SEEDS,cells_from_initial,bootstrap

THRESHOLDS=(0.5,1.0,1.5,2.0)

def matched_mean(delta,mask,cells):
    count=int(mask.sum())
    assert count>0
    weighted=0.
    for cell in np.unique(cells):
        pool=cells==cell
        weighted+=int(np.sum(mask & pool))*float(np.mean(delta[pool]))
    return weighted/count

def main():
    out=ROOT/'results/cpa_thresholds_20260916';out.mkdir(parents=True,exist_ok=True)
    rows=[];hashes={};support=[]
    for ti,task in enumerate(TASKS[:-1]):
        path=CACHE/f'{task}__test.npz'
        hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as a:
            distance=a['risk_distance'];runs=a['run_index']
        masks={alpha:distance<=alpha for alpha in THRESHOLDS}
        assert all(np.all(~masks[a] | masks[b]) for a,b in zip(THRESHOLDS[:-1],THRESHOLDS[1:]))
        for alpha,mask in masks.items():
            support.append(dict(task=task,threshold=alpha,n=len(mask),cpa_n=int(mask.sum()),fraction=float(mask.mean())))
        for method in METHODS:
            for seed in SEEDS:
                paths=[ROOT/f'results/synthetic-seed{seed}/predictions/{method}__seed{seed}__step{step}__{task}.npz' for step in [ti,5]]
                values=[]
                for p in paths:
                    hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
                    with np.load(p) as a:values.append(a['fde'].astype(float))
                first,last=values;delta=last-first
                assert len(delta)==len(distance) and np.isfinite(delta).all()
                cells=cells_from_initial(runs,first)
                for alpha,mask in masks.items():
                    cpa=float(delta[mask].mean());overall=float(delta.mean());matched=matched_mean(delta,mask,cells)
                    rows.append(dict(method=method,seed=seed,task=task,threshold=alpha,
                        overall=overall,cpa=cpa,matched=matched,cpa_minus_overall=cpa-overall,cpa_minus_matched=cpa-matched))
    frame=pd.DataFrame(rows)
    assert len(frame)==800
    frame.to_csv(out/'per_task.csv',index=False)
    metrics=['overall','cpa','matched','cpa_minus_overall','cpa_minus_matched']
    per_seed=frame.groupby(['method','seed','threshold'])[metrics].mean().reset_index()
    per_seed.to_csv(out/'per_seed.csv',index=False)
    rows=[]
    for mi,method in enumerate(METHODS):
        for alpha in THRESHOLDS:
            part=per_seed[(per_seed.method==method)&(per_seed.threshold==alpha)].sort_values('seed')
            assert len(part)==8
            for metric in metrics:
                mean,lo,hi=bootstrap(part[metric],np.random.default_rng(613+mi))
                rows.append(dict(method=method,threshold=alpha,metric=metric,n_seeds=8,mean=mean,ci_low=lo,ci_high=hi))
    summary=pd.DataFrame(rows);summary.to_csv(out/'summary.csv',index=False)
    pd.DataFrame(support).to_csv(out/'support.csv',index=False)
    reference=pd.read_csv(ROOT/'results/stratification/summary.csv')
    for row in summary[summary.threshold==1].itertuples():
        ref=reference[(reference.method==row.method)&(reference.metric==row.metric)].iloc[0]
        np.testing.assert_allclose([row.mean,row.ci_low,row.ci_high],[ref['mean'],ref.ci_low,ref.ci_high],rtol=0,atol=1e-12)
    assert matched_mean(np.zeros(6),np.array([1,0,1,0,0,1],bool),np.array([0,0,0,1,1,1]))==0
    (out/'audit.json').write_text(json.dumps({'analysis':'Post-primary sensitivity fixed on 2026-09-16 before computation',
      'thresholds':THRESHOLDS,'historical_tasks':TASKS[:-1],'seeds':SEEDS,'methods':METHODS,
      'source_files':len(hashes),'source_sha256':hashes,'task_rows':len(frame),
      'main_threshold_reproduces_all_original_means_and_intervals':True,
      'uniform_scale_equivalence':'threshold alpha corresponds to axes 50*alpha,50*alpha,15*alpha meters; their aspect ratio is unchanged',
      'null_check':True,'nested_masks':True,'bootstrap':'20,000 paired training-seed draws, conditional on fixed histories, tasks and windows; descriptive intervals, no multiplicity correction',
      'matching':'Exact expectation; run and initial-error deciles fixed across thresholds; no final error used in membership or matching'},indent=2))
    print(summary[(summary.method=='reservoir')&(summary.metric=='cpa_minus_matched')].to_string(index=False))

if __name__=='__main__':main()
