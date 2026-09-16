"""TTC-style time-to-ellipsoid-entry controls on frozen RIFT predictions.

`prepare` reads observation-time states only. `analyze` reuses saved endpoint
errors. `check` verifies geometry without datasets. Raw window data stay local.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from analyze_stratification import ROOT,CACHE,TASKS,METHODS,SEEDS,cells_from_initial,bootstrap

DATA=ROOT/'data/ttc_control'
OUT=ROOT/'results/ttc_control_20260916'
MX=math.cos(math.radians(4.62))*111320.
MY=110540.
STRATA=('within_3s','within_5s','within_10s','ongoing','new_0_3s','new_3_5s','new_5_10s','no_entry_10s')
BANDS=STRATA[3:]
INSIDE_BANDS=BANDS[:-1]
METRICS=('overall','stratum','matched','stratum_minus_overall','stratum_minus_matched',
         'cpa_matched','stratum_minus_cpa_matched')


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def entry_time(position,velocity):
    """First t>=0 with ||position+t*velocity||<=1; inf if none.

    The rationalized smaller quadratic root avoids cancellation near the
    boundary. Position and velocity are already scaled by ellipsoid axes.
    """
    a=np.asarray(position,dtype=np.float64)
    b=np.asarray(velocity,dtype=np.float64)
    aa=np.sum(b*b,axis=-1)
    ab=np.sum(a*b,axis=-1)
    cc=np.sum(a*a,axis=-1)-1.
    disc=ab*ab-aa*cc
    out=np.full(cc.shape,np.inf)
    inside=cc<=0.
    out[inside]=0.
    enters=(~inside)&(aa>0.)&(ab<0.)&(disc>=0.)
    out[enters]=cc[enters]/(-ab[enters]+np.sqrt(disc[enters]))
    return out


def masks_from_entry(t):
    return dict(zip(STRATA,[t<=3,t<=5,t<=10,t==0,(t>0)&(t<=3),
                           (t>3)&(t<=5),(t>5)&(t<=10),t>10]))


def check_geometry():
    pos=np.array([[2,0,0],[2,0,0],[2,0,0],[.5,0,0],[-2,1,0],[0,0,2]],float)
    vel=np.array([[-.5,0,0],[.5,0,0],[0,0,0],[1,0,0],[1,0,0],[0,0,-.25]],float)
    np.testing.assert_allclose(entry_time(pos,vel),[2,np.inf,np.inf,0,2,4],rtol=0,atol=1e-12)
    rng=np.random.default_rng(20260916)
    a=rng.normal(size=(50000,3))*3
    b=rng.normal(size=(50000,3))
    b[::29]=0
    t=entry_time(a,b)
    aa=np.sum(b*b,axis=1)
    for horizon in (3.,5.,10.):
        tc=np.clip(-np.sum(a*b,axis=1)/np.maximum(aa,1e-30),0,horizon)
        cpa=np.linalg.norm(a+tc[:,None]*b,axis=1)
        np.testing.assert_array_equal(t<=horizon,cpa<=1)
    new=np.isfinite(t)&(t>0)
    np.testing.assert_allclose(np.linalg.norm(a[new]+t[new,None]*b[new],axis=1),1.,rtol=0,atol=1e-11)
    assert np.all(np.linalg.norm(a[new]+(.999*t[new,None])*b[new],axis=1)>1)
    masks=masks_from_entry(t)
    assert np.all(np.sum([masks[k] for k in BANDS],axis=0)==1)
    assert np.array_equal(np.any([masks[k] for k in INSIDE_BANDS],axis=0),masks['within_10s'])
    return {'analytic_examples':6,'random_kinematic_pairs':len(a),
            'three_horizon_cpa_equivalence':True,'boundary_root_checks':True}


def prepare():
    DATA.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    geometry=check_geometry();hashes={};audits=[];support=[]
    for task in TASKS[:-1]:
        cache=CACHE/f'{task}__test.npz'
        hashes[cache.relative_to(ROOT).as_posix()]=sha256(cache)
        with np.load(cache) as data:
            meta=data['meta'];runs=data['run_index']
            original_cpa=data['risk_distance'];original_current=data['current_distance']
        n=len(meta);times_out=np.full(n,np.nan)
        cpa_out=np.full(n,np.nan,np.float32);current_out=np.full(n,np.nan,np.float32)
        count_out=np.zeros(n,np.int16)
        paths=sorted((ROOT/'data/lowaltitude/states').glob(f'{task}__*.parquet'))
        assert len(paths)==5
        for ri,path in enumerate(paths):
            hashes[path.relative_to(ROOT).as_posix()]=sha256(path)
            frame=pd.read_parquet(path,columns=['sim_time_s','drone_id','lat_deg','lon_deg','alt_m','gs_ms','vs_ms','trk_deg'])
            groups=frame.groupby('sim_time_s',sort=False).indices
            indices=np.flatnonzero(runs==ri)
            for time_s in np.unique(meta[indices,1].astype(int)):
                qi=indices[meta[indices,1].astype(int)==time_s]
                snapshot=frame.iloc[groups[time_s]]
                assert len(snapshot)>=2
                xy=np.column_stack(((snapshot.lon_deg.to_numpy()+74.10)*MX,
                                    (snapshot.lat_deg.to_numpy()-4.62)*MY)).astype(np.float32)
                xyz=np.column_stack((xy[:,0]/50.,xy[:,1]/50.,snapshot.alt_m.to_numpy()/15.))
                track=np.deg2rad(snapshot.trk_deg.to_numpy());speed=snapshot.gs_ms.to_numpy()
                velocity=np.column_stack((speed*np.sin(track)/50.,speed*np.cos(track)/50.,
                                          snapshot.vs_ms.to_numpy()/15.)).astype(np.float32)
                rows=meta[qi]
                query=np.column_stack((rows[:,2]/50.,rows[:,3]/50.,rows[:,4]/15.))
                distances,neighbors=cKDTree(xyz).query(query,k=min(64,len(snapshot)),workers=1)
                ids={int(d):i for i,d in enumerate(snapshot.drone_id.to_numpy())}
                self_index=np.array([ids[int(row[0])] for row in rows])
                self_mask=neighbors==self_index[:,None]
                relp=xyz[neighbors]-query[:,None,:]
                relv=velocity[neighbors]-velocity[self_index][:,None,:]
                # Original mixed-precision calculation for cache reconstruction.
                aa=np.sum(relv*relv,axis=2)
                tc=np.clip(-np.sum(relp*relv,axis=2)/np.maximum(aa,1e-12),0.,10.)
                cpa=np.linalg.norm(relp+tc[:,:,None]*relv,axis=2)
                cpa[self_mask]=np.inf;distances[self_mask]=np.inf
                entry=entry_time(relp,relv);entry[self_mask]=np.inf
                times_out[qi]=np.min(entry,axis=1)
                cpa_out[qi]=np.min(cpa,axis=1).astype(np.float32)
                current_out[qi]=np.min(distances,axis=1).astype(np.float32)
                count_out[qi]=np.sum(~self_mask,axis=1)
            print('prepared',path.name,flush=True)
        assert not np.isnan(times_out).any()
        np.testing.assert_allclose(cpa_out,original_cpa,rtol=2e-6,atol=2e-5)
        np.testing.assert_allclose(current_out,original_current,rtol=2e-6,atol=2e-5)
        masks=masks_from_entry(times_out)
        np.testing.assert_array_equal(cpa_out<=1,original_cpa<=1)
        np.testing.assert_array_equal(masks['within_10s'],original_cpa<=1)
        assert np.all(np.sum([masks[k] for k in BANDS],axis=0)==1)
        assert np.all(~masks['within_3s']|masks['within_5s'])
        assert np.all(~masks['within_5s']|masks['within_10s'])
        np.savez_compressed(DATA/f'{task}.npz',entry_time=times_out,cpa_distance=cpa_out,
                            current_distance=current_out,neighbor_count=count_out)
        for stratum,mask in masks.items():
            support.append(dict(task=task,stratum=stratum,n=n,stratum_n=int(mask.sum()),
                                fraction=float(mask.mean()),overlap_cpa_n=int(np.sum(mask&(original_cpa<=1)))))
        audits.append(dict(task=task,windows=n,max_cpa_distance_difference=float(np.max(np.abs(cpa_out-original_cpa))),
                           max_current_distance_difference=float(np.max(np.abs(current_out-original_current))),
                           min_other_neighbors=int(count_out.min()),max_other_neighbors=int(count_out.max()),
                           cpa_mask_exact=True,tte_10s_equals_cpa_exact=True))
    pd.DataFrame(support).to_csv(OUT/'support.csv',index=False)
    (OUT/'geometry_audit.json').write_text(json.dumps({'design_date':'2026-09-16','geometry':geometry,
        'tasks':audits,'source_sha256':hashes,'inputs':'Only observation-time snapshots; no recorded futures used',
        'query':'k=min(64,snapshot size), then focal aircraft excluded; matches original cache'},indent=2)+'\n')
    print(pd.DataFrame(support).pivot(index='task',columns='stratum',values='stratum_n').to_string())


def conditional_mean(delta,mask,cells,pool=None):
    if pool is None:pool=np.ones(len(delta),bool)
    assert np.all(~mask|pool)
    count=int(mask.sum())
    if not count:return np.nan
    value=0.
    for c in np.unique(cells[mask]):
        take=(cells==c)&pool
        value+=int(np.sum(mask&(cells==c)))*float(delta[take].mean())
    return value/count


def analyze():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[];hashes={};mixture_error=0.
    for ti,task in enumerate(TASKS[:-1]):
        data_path=DATA/f'{task}.npz'
        hashes[data_path.relative_to(ROOT).as_posix()]=sha256(data_path)
        with np.load(data_path) as a:entry=a['entry_time']
        with np.load(CACHE/f'{task}__test.npz') as a:runs=a['run_index'];cpa=a['risk_distance']<=1
        masks=masks_from_entry(entry)
        np.testing.assert_array_equal(masks['within_10s'],cpa)
        for method in METHODS:
            for seed in SEEDS:
                values=[]
                for step in [ti,5]:
                    path=ROOT/f'results/synthetic-seed{seed}/predictions/{method}__seed{seed}__step{step}__{task}.npz'
                    hashes[path.relative_to(ROOT).as_posix()]=sha256(path)
                    with np.load(path) as a:values.append(a['fde'].astype(float))
                first,last=values;delta=last-first;cells=cells_from_initial(runs,first)
                assert len(delta)==len(entry) and np.isfinite(delta).all()
                for name,mask in masks.items():
                    count=int(mask.sum())
                    group=float(delta[mask].mean()) if count else np.nan
                    ordinary=float(delta.mean());matched=conditional_mean(delta,mask,cells)
                    inner=conditional_mean(delta,mask,cells,pool=cpa) if name in INSIDE_BANDS else np.nan
                    rows.append(dict(method=method,seed=seed,task=task,stratum_name=name,stratum_n=count,
                        overall=ordinary,stratum=group,matched=matched,stratum_minus_overall=group-ordinary,
                        stratum_minus_matched=group-matched,cpa_matched=inner,stratum_minus_cpa_matched=group-inner))
                for names,pool in [(BANDS,np.ones(len(delta),bool)),(INSIDE_BANDS,cpa)]:
                    total=sum(float(delta[masks[k]].sum()) for k in names)/int(pool.sum())
                    error=abs(total-float(delta[pool].mean()));mixture_error=max(mixture_error,error)
                    assert error<1e-12
    frame=pd.DataFrame(rows)
    assert len(frame)==1600
    frame.to_csv(OUT/'per_task.csv',index=False)
    # Do not let pandas silently omit empty task/group cells.
    grouped=frame.groupby(['method','seed','stratum_name'])[list(METRICS)]
    per_seed=grouped.agg(lambda x:float(np.mean(x.to_numpy()))).reset_index()
    per_seed.to_csv(OUT/'per_seed.csv',index=False)
    rows=[]
    for mi,method in enumerate(METHODS):
        for name in STRATA:
            part=per_seed[(per_seed.method==method)&(per_seed.stratum_name==name)].sort_values('seed')
            assert len(part)==8
            for metric in METRICS:
                vals=part[metric].to_numpy();finite=int(np.isfinite(vals).sum())
                result=bootstrap(vals,np.random.default_rng(613+mi)) if finite==8 else (np.nan,np.nan,np.nan)
                rows.append(dict(method=method,stratum_name=name,metric=metric,n_seeds=finite,
                                 mean=result[0],ci_low=result[1],ci_high=result[2]))
    summary=pd.DataFrame(rows);summary.to_csv(OUT/'summary.csv',index=False)
    original=pd.read_csv(ROOT/'results/stratification/summary.csv')
    metric_map={'overall':'overall','stratum':'cpa','matched':'matched',
                'stratum_minus_overall':'cpa_minus_overall','stratum_minus_matched':'cpa_minus_matched'}
    for row in summary[summary.stratum_name=='within_10s'].itertuples():
        if row.metric not in metric_map:continue
        ref=original[(original.method==row.method)&(original.metric==metric_map[row.metric])].iloc[0]
        np.testing.assert_allclose([row.mean,row.ci_low,row.ci_high],[ref['mean'],ref.ci_low,ref.ci_high],rtol=0,atol=1e-12)
    assert conditional_mean(np.zeros(5),np.array([1,0,0,1,0],bool),np.array([0,0,1,1,1]))==0
    (OUT/'analysis_audit.json').write_text(json.dumps({'design':'Post-primary 2026-09-16; original results known',
        'methods':METHODS,'seeds':SEEDS,'tasks':TASKS[:-1],'strata':STRATA,'primary_temporal_contrast':'new_0_3s',
        'per_task_rows':len(frame),'per_seed_rows':len(per_seed),'summary_rows':len(summary),
        'empty_task_group_cells':int(np.sum(frame.stratum_n==0)),
        'undefined_cpa_conditional_cells':'Expected outside the four ongoing/new-entry bands; no conditional estimate is asserted there',
        'max_count_weighted_reconstruction_error_m':mixture_error,'tte_10s_reproduces_primary_intervals':True,
        'zero_change_control':True,'source_sha256':hashes,
        'uncertainty':'20,000 training-seed bootstrap draws, conditional on fixed task order/data; no multiplicity correction'},indent=2)+'\n')
    print(summary[(summary.method=='reservoir')&summary.metric.isin(['stratum_minus_matched','stratum_minus_cpa_matched'])].to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['check','prepare','analyze'])
    args=parser.parse_args()
    if args.stage=='check':print(json.dumps(check_geometry(),indent=2))
    elif args.stage=='prepare':prepare()
    else:analyze()
