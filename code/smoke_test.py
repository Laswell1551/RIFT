"""Small artificial-input software checks; these values are not paper evidence."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import tempfile
import numpy as np
import pandas as pd
from analyze_stratification import cells_from_initial,control

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--road',action='store_true',help='Also check the separately fetched road encoder.')
    args=parser.parse_args()
    cells=cells_from_initial(np.repeat([0,1],40),np.tile(np.arange(40),2))
    mask=np.arange(80)%3==0
    mean,draws,indices,counts=control(np.zeros(80),mask,cells,np.random.default_rng(4),draws=20)
    assert mean==0 and not draws.any() and len(set(indices))==mask.sum()
    assert all((cells[indices]==c).sum()==((cells==c)&mask).sum() for c in np.unique(cells))
    with tempfile.TemporaryDirectory(prefix='rift-smoke-') as tmp:
        folder=Path(tmp);cache=folder/'cache';cache.mkdir()
        rng=np.random.default_rng(810)
        for task in ['first','second']:
            for split in ['train','test']:
                n=24
                obs=rng.normal(size=(n,5,3)).astype('float32')
                fut=rng.normal(size=(n,10,3)).astype('float32')
                np.savez_compressed(cache/f'{task}__{split}.npz',obs=obs,future=fut,
                    context=np.ones((n,6),np.float32),risk_score=np.linspace(.1,.9,n,dtype='float32'),
                    risk_distance=np.linspace(.1,2,n,dtype='float32'))
        cfg=json.loads((ROOT/'configs/main_13_methods.json').read_text())
        cfg.update(task_order=['first','second'],epochs_per_task=1,batch_size=8,memory_budget=8,
                   train_limit_per_task=24,eval_limit_per_task=24,torch_threads=1)
        (folder/'config.json').write_text(json.dumps(cfg))
        process=subprocess.run([sys.executable,str(ROOT/'code/run_continual.py'),'--config',str(folder/'config.json'),
             '--cache',str(cache),'--output',str(folder/'output'),'--seed','7'],capture_output=True,text=True)
        if process.returncode:raise RuntimeError(process.stdout+'\n'+process.stderr)
        for method in cfg['methods']:
            f=pd.read_csv(folder/f'output/metrics__{method}__seed7.csv')
            assert len(f)==3 and np.isfinite(f[['fde','risk_fde']]).all().all()
            assert (f.memory_size<=8).all()
            before=np.load(folder/f'output/predictions/{method}__seed7__step0__first.npz')['prediction']
            after=np.load(folder/f'output/predictions/{method}__seed7__step1__first.npz')['prediction']
            assert not np.allclose(before,after),f'No historical forecast change: {method}'
    if args.road:
        import torch
        from run_interaction import EndpointPredictor
        torch.set_num_threads(1);torch.manual_seed(46)
        net=EndpointPredictor(64).eval()
        assert sum(p.numel() for p in net.parameters() if p.requires_grad)==165570
        # Valid target + one neighbor and three map polylines; the rest are padding.
        valid=torch.zeros(2,81);valid[:,:3]=1;valid[:,55:57]=1
        data={'trajectory':torch.randn(2,26,9,8),'maps':torch.randn(2,55,5,5),
              'lane':torch.randn(2,55,5),'af':torch.eye(55).repeat(2,1,1),'valid':valid}
        with torch.no_grad(): before=net(data,np.arange(2))
        changed={k:v.clone() for k,v in data.items()}
        changed['trajectory'][:,2:]=1234;changed['maps'][:,3:]=4321;changed['lane'][:,3:]=5432
        with torch.no_grad():after=net(changed,np.arange(2))
        torch.testing.assert_close(before,after,atol=1e-6,rtol=1e-6)
        changed={k:v.clone() for k,v in data.items()};changed['trajectory'][:,1,-1,2:6]+=20
        with torch.no_grad():moved=net(changed,np.arange(2))
        assert not torch.allclose(before,moved,atol=1e-6)
        optimizer=torch.optim.AdamW(net.parameters(),lr=.001)
        optimizer.zero_grad();net(data,np.arange(2)).square().mean().backward();optimizer.step()
        with torch.no_grad():updated=net(data,np.arange(2))
        assert torch.isfinite(updated).all() and not torch.allclose(before,updated)
    print(json.dumps({'no_change_control_and_matching_counts':'PASS',
      'thirteen_methods_two_updates_historical_backtest':'PASS','memory_budget':'PASS',
      'road_mask_neighbor_sensitivity_parameter_update':'PASS' if args.road else 'not requested'},indent=2))

if __name__=='__main__':main()
