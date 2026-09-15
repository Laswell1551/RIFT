"""End-to-end road domain updates with a public UQnet encoder regression adaptation."""
from pathlib import Path
import argparse
import hashlib
import json
import random
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from uqnet_vendor.encoder import VectorEncoder
from memory import ReplayMemory, CGSMDualMemory
from analyze_stratification import ROOT


class EndpointPredictor(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.encoder = VectorEncoder({'encoder_attention_size': width, 'inference': True})
        self.hidden = nn.Sequential(nn.Linear(width, width), nn.ReLU())
        self.output = nn.Linear(width, 2)

    def forward(self, data, indices, features=False):
        t, m, l, af, valid = [data[k][indices] for k in ['trajectory','maps','lane','af','valid']]
        adj = valid[:,:,None] * valid[:,None,:]
        h = self.encoder(m, t, l, adj, af, valid)[2][:,55]
        h = self.hidden(h)
        pred = self.output(h) + t[:,0,-1,4:6] * .1
        return (pred, h) if features else pred


def load_all(cfg):
    cache = ROOT / 'data/interaction_cache'
    blocks = {'train': [], 'val': []}
    slices = {'train': [], 'val': []}
    for split in blocks:
        offset = 0
        for task in cfg['task_order']:
            with np.load(cache / f'{task}__{split}.npz') as a:
                block = {k: a[k] for k in a.files}
            blocks[split].append(block)
            slices[split].append(np.arange(offset, offset + len(block['target'])))
            offset += len(block['target'])
    data = {}
    for split in blocks:
        joined = {k: np.concatenate([b[k] for b in blocks[split]]) for k in blocks[split][0]}
        data[split] = {k: torch.from_numpy(joined[k].astype(np.float32)) for k in ['trajectory','maps','lane','af','valid','target']}
        data[split].update({k: joined[k] for k in ['cpa','observable','source_index']})
    return data, slices


@torch.no_grad()
def predict(model, data, ids, features=False):
    model.eval()
    ps, hs = [], []
    for start in range(0, len(ids), 128):
        out = model(data, ids[start:start+128], features=features)
        if features:
            p, h = out
            hs.append(h.numpy())
        else:
            p = out
        ps.append(p.numpy())
    return (np.concatenate(ps), np.concatenate(hs)) if features else np.concatenate(ps)


def payload(model, data, ids, ti, cfg, projection):
    p, h = predict(model, data, ids, features=True)
    y = data['target'][ids].numpy() / 30
    errors = p - y
    gradient = (errors[:,:,None] * np.c_[h,np.ones(len(h))][:,None,:]).reshape(len(h),-1)
    magnitude = np.linalg.norm(gradient, axis=1)
    embedding = gradient @ projection
    embedding /= np.maximum(np.linalg.norm(embedding,axis=1,keepdims=True),1e-8)
    cpa = data['cpa'][ids]
    return {'inputs': ids[:,None], 'targets': y, 'future': y[:,None]*30,
        'risk_score': np.exp(-cpa/8), 'risk_distance': cpa/4,
        'task_index': np.full(len(ids),ti), 'logits': p,
        'selection_embedding': embedding.astype(np.float32), 'observable_embedding': data['observable'][ids],
        'selection_magnitude': magnitude, 'replay_priority': np.zeros(len(ids),np.float32)}


def run(method, seed, cfg, data, slices):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = EndpointPredictor(cfg['encoder_attention_size'])
    rng = np.random.default_rng(seed + 9000)
    projection = np.random.default_rng(seed + 2026).choice([-1.,1.],size=(2*(cfg['encoder_attention_size']+1),cfg['gradient_projection_dim'])).astype(np.float32) / np.sqrt(cfg['gradient_projection_dim'])
    memory = CGSMDualMemory(cfg['memory_budget'],seed,cfg['risk_power'],cfg['general_fraction']) if method == 'cgsm_dual' else ReplayMemory(cfg['memory_budget'],method,seed)
    out = ROOT / f'results/interaction-seed{seed}/{method}'
    out.mkdir(parents=True,exist_ok=True)
    (out/'predictions').mkdir(exist_ok=True)
    metadata = dict(cfg, method=method, seed=seed, parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
        torch_version=torch.__version__, started_utc=pd.Timestamp.now(tz='UTC').isoformat(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'run_config.json').write_text(json.dumps(metadata,indent=2))
    rows, histories = [], []
    for ti, task in enumerate(cfg['task_order']):
        ids = slices['train'][ti]
        optimizer = torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        start_time = time.perf_counter()
        for epoch in range(cfg['epochs_per_task']):
            model.train()
            order = rng.permutation(ids)
            losses = []
            for start in range(0,len(order),cfg['batch_size']):
                current = order[start:start+cfg['batch_size']]
                batch = current
                stored = memory.data
                if stored is not None:
                    if method == 'cgsm_dual':
                        is_safety = stored['replay_priority'] > .5
                        prob = np.where(is_safety, cfg['safety_replay_fraction']/max(is_safety.sum(),1),
                            (1-cfg['safety_replay_fraction'])/max((~is_safety).sum(),1))
                        prob = prob/prob.sum()
                        sample = rng.choice(len(stored['inputs']),len(current),replace=True,p=prob)
                    else:
                        sample = rng.integers(0,len(stored['inputs']),len(current))
                    replay = stored['inputs'][sample,0].astype(np.int64)
                    assert np.all(replay < ids[0]), 'Replay must reference earlier training domains only.'
                    batch = np.r_[current,replay]
                optimizer.zero_grad(set_to_none=True)
                prediction = model(data['train'],batch)
                loss = nn.functional.smooth_l1_loss(prediction,data['train']['target'][batch]/30,beta=.1)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'])
                optimizer.step()
                losses.append(float(loss.detach()))
            histories.append({'step':ti,'task':task,'epoch':epoch,'train_loss':float(np.mean(losses)), 'optimizer_steps':len(losses)})
            print(method,seed,task,'epoch',epoch,'loss',np.mean(losses),flush=True)
        seconds = time.perf_counter()-start_time
        torch.save(model.state_dict(),out/f'checkpoint_step{ti}.pt')
        for ei in range(ti+1):
            eid = slices['val'][ei]
            p = predict(model,data['val'],eid)*30
            fde = np.linalg.norm(p-data['val']['target'][eid].numpy(),axis=1)
            cpa = data['val']['cpa'][eid] < cfg['cpa_threshold_m']
            rows.append({'method':method,'seed':seed,'learn_step':ti,'learn_task':task,'eval_task_index':ei,
                'eval_task':cfg['task_order'][ei],'n':len(eid),'cpa_n':int(cpa.sum()),'fde':float(fde.mean()),
                'risk_fde':float(fde[cpa].mean()) if cpa.any() else np.nan,'train_seconds':seconds})
            if ei==ti or ti==len(cfg['task_order'])-1:
                np.savez_compressed(out/f'predictions/step{ti}__{cfg["task_order"][ei]}.npz', prediction=p,fde=fde,source_index=data['val']['source_index'][eid])
        pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
        pd.DataFrame(histories).to_csv(out/'training_history.csv',index=False)
        if method!='finetune' and ti<len(cfg['task_order'])-1:
            memory.update(payload(model,data['train'],ids,ti,cfg,projection))
            np.savez_compressed(out/f'memory_step{ti}.npz',ids=memory.data['inputs'][:,0],task_index=memory.data['task_index'],branch=memory.data['replay_priority'])
        print('EVALUATED',method,seed,task,'training seconds',round(seconds,2),flush=True)
    (out/'COMPLETE.json').write_text(json.dumps({'completed_utc':pd.Timestamp.now(tz='UTC').isoformat(),'rows':len(rows)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--method',required=True)
    args=parser.parse_args()
    cfg=json.loads((ROOT/'configs/interaction.json').read_text())
    torch.set_num_threads(cfg['torch_threads'])
    data,slices=load_all(cfg)
    run(args.method,args.seed,cfg,data,slices)
