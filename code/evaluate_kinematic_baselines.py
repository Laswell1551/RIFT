from pathlib import Path
import numpy as np
import pandas as pd
root=Path(__file__).resolve().parents[2]
cache=root/'data'/'cache'/'cgsm-final-locked-3d-cpa'
tasks=('baseline','wind5','speed15','alt180','speed10','alt200')
rows=[]
for task in tasks:
    a=np.load(cache/f'{task}__test.npz')
    obs=a['obs']; fut=a['future']; risk=a['risk_distance']<=1
    v=obs[:,-1]-obs[:,-2]
    acc=obs[:,-1]-2*obs[:,-2]+obs[:,-3]
    v_ca=(obs[:,-1]-obs[:,-2])+0.5*acc
    k=np.arange(1,fut.shape[1]+1,dtype=np.float32)[None,:,None]
    preds={'Constant velocity':obs[:,-1:, :]+v[:,None,:]*k,
           'Constant acceleration':obs[:,-1:,:]+v_ca[:,None,:]*k+0.5*acc[:,None,:]*k**2}
    for method,pred in preds.items():
        d=np.linalg.norm(pred-fut,axis=2)
        rows.append(dict(task=task,method=method,n=len(d),risk_n=int(risk.sum()),ade=d.mean(),fde=d[:,-1].mean(),risk_ade=d[risk].mean(),risk_fde=d[risk,-1].mean()))
df=pd.DataFrame(rows)
out=root/'experiments'/'results'/'tits-report-extension-aggregate'/'kinematic_baselines.csv'
df.to_csv(out,index=False)
for method,q in df.groupby('method'):
    print(method, 'ADE', np.average(q.ade,weights=q.n), 'FDE',np.average(q.fde,weights=q.n),'risk ADE',np.average(q.risk_ade,weights=q.risk_n),'risk FDE',np.average(q.risk_fde,weights=q.risk_n))