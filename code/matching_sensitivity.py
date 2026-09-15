import json
import numpy as np
import pandas as pd
from analyze_stratification import ROOT,CACHE,TASKS,METHODS,SEEDS,bootstrap

def main():
    rows=[]; audits={}
    for method in METHODS:
        for seed in SEEDS:
            for ti,task in enumerate(TASKS[:-1]):
                a=np.load(CACHE/f'{task}__test.npz')
                initial=np.load(ROOT/f'results/synthetic-seed{seed}/predictions/{method}__seed{seed}__step{ti}__{task}.npz')['fde'].astype(float)
                final=np.load(ROOT/f'results/synthetic-seed{seed}/predictions/{method}__seed{seed}__step5__{task}.npz')['fde'].astype(float)
                delta=final-initial;mask=a['risk_distance']<=1;run=a['run_index']
                cpa=np.flatnonzero(mask);donors=np.empty(len(cpa),int)
                cells=np.empty(len(mask),int)
                for r in np.unique(run):
                    ix=np.flatnonzero(run==r)
                    cuts=np.unique(np.quantile(initial[ix],np.arange(1,50)/50))
                    cells[ix]=int(r)*50+np.searchsorted(cuts,initial[ix],side='right')
                    target_positions=np.flatnonzero(run[cpa]==r)
                    controls=np.flatnonzero((run==r)&~mask)
                    assert len(controls)>0
                    donors[target_positions]=controls[np.abs(initial[cpa[target_positions],None]-initial[controls][None,:]).argmin(axis=1)]
                expected=sum(mask[cells==c].sum()*delta[cells==c].mean() for c in np.unique(cells))/mask.sum()
                expected_initial=sum(mask[cells==c].sum()*initial[cells==c].mean() for c in np.unique(cells))/mask.sum()
                distance=np.abs(initial[cpa]-initial[donors])
                rows.append({'method':method,'seed':seed,'task':task,'cpa_minus_50bin':delta[mask].mean()-expected,
                    'initial_50bin_residual':initial[mask].mean()-expected_initial,
                    'cpa_minus_nearest_non_cpa':np.mean(delta[cpa]-delta[donors]),
                    'initial_nearest_residual':np.mean(initial[cpa]-initial[donors]),
                    'nearest_initial_mae':distance.mean(),'nearest_initial_max':distance.max(),
                    'cpa_n':len(cpa),'unique_donors':len(np.unique(donors))})
                audits[f'{method}__{seed}__{task}']={'cpa_indices':cpa.tolist(),'donor_indices':donors.tolist(),'fifty_bin_cells':cells.tolist()}
    f=pd.DataFrame(rows);out=ROOT/'results/stratification'
    f.to_csv(out/'matching_sensitivity_per_task.csv',index=False)
    s=f.groupby(['method','seed']).mean(numeric_only=True).reset_index()
    s.to_csv(out/'matching_sensitivity_per_seed.csv',index=False)
    result=[]
    for mi,method in enumerate(METHODS):
        for metric in ['cpa_minus_50bin','cpa_minus_nearest_non_cpa','initial_50bin_residual','initial_nearest_residual','nearest_initial_mae']:
            mean,lo,hi=bootstrap(s[s.method==method][metric],np.random.default_rng(830+mi))
            result.append({'method':method,'metric':metric,'mean':mean,'ci_low':lo,'ci_high':hi})
    pd.DataFrame(result).to_csv(out/'matching_sensitivity_summary.csv',index=False)
    (out/'matching_sensitivity_audit.json').write_text(json.dumps(audits))
    print(pd.DataFrame(result).query("metric.str.startswith('cpa_')",engine='python').to_string(index=False))

if __name__=='__main__':
    main()
