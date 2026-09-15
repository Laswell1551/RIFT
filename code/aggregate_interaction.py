"""Require every declared road run, then aggregate paired historical-domain retention."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from analyze_stratification import ROOT, bootstrap

def main():
    cfg=json.loads((ROOT/'configs/interaction.json').read_text())
    out=ROOT/'results/interaction_summary'
    out.mkdir(exist_ok=True)
    all_rows=[]
    values=[]
    domain=[]
    for mi,method in enumerate(cfg['methods']):
        for seed in cfg['seeds']:
            folder=ROOT/f'results/interaction-seed{seed}/{method}'
            assert (folder/'COMPLETE.json').exists(), f'Incomplete run: {method} {seed}'
            f=pd.read_csv(folder/'metrics.csv')
            assert len(f)==36 and not f[['fde','risk_fde']].isna().any().any()
            assert len(f[['learn_step','eval_task_index']].drop_duplicates())==36
            assert all((f.learn_step==i).sum()==i+1 for i in range(8))
            all_rows.append(f)
            final=f[f.learn_step==7].sort_values('eval_task_index')
            diag=f[f.learn_step==f.eval_task_index].sort_values('eval_task_index')
            forgetting=final[['fde','risk_fde']].to_numpy()-diag[['fde','risk_fde']].to_numpy()
            value={'method':method,'seed':seed,'final_fde':final.fde.mean(),'final_cpa_fde':final.risk_fde.mean(),
                'ordinary_forgetting':forgetting[:-1,0].mean(),'cpa_forgetting':forgetting[:-1,1].mean(),
                'gap':(forgetting[:-1,1]-forgetting[:-1,0]).mean()}
            values.append(value)
            for ti,task in enumerate(cfg['task_order']):
                domain.append({'method':method,'seed':seed,'task':task,'historical':ti<7,
                    'n':int(final.iloc[ti].n),'cpa_n':int(final.iloc[ti].cpa_n),
                    'initial_fde':diag.iloc[ti].fde,'final_fde':final.iloc[ti].fde,
                    'initial_cpa_fde':diag.iloc[ti].risk_fde,'final_cpa_fde':final.iloc[ti].risk_fde,
                    'ordinary_forgetting':forgetting[ti,0],'cpa_forgetting':forgetting[ti,1],
                    'gap':forgetting[ti,1]-forgetting[ti,0]})
    byseed=pd.DataFrame(values)
    byseed.to_csv(out/'per_seed.csv',index=False)
    pd.concat(all_rows).to_csv(out/'all_checkpoint_metrics.csv',index=False)
    pd.DataFrame(domain).to_csv(out/'per_domain.csv',index=False)
    rows=[]
    for mi,method in enumerate(cfg['methods']):
        for metric in ['final_fde','final_cpa_fde','ordinary_forgetting','cpa_forgetting','gap']:
            v=byseed[byseed.method==method].sort_values('seed')[metric].to_numpy()
            mean,lo,hi=bootstrap(v,np.random.default_rng(456+mi))
            rows.append({'method':method,'metric':metric,'n_seeds':len(v),'mean':mean,'sd':v.std(ddof=1),'ci_low':lo,'ci_high':hi})
    result=pd.DataFrame(rows)
    result.to_csv(out/'summary.csv',index=False)
    print(result.to_string(index=False))

if __name__=='__main__':
    main()
