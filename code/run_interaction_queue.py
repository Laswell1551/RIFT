from pathlib import Path
import concurrent.futures
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
(ROOT/"logs").mkdir(exist_ok=True)
cfg=json.loads((ROOT/'configs/interaction.json').read_text())
jobs=[(seed,method) for seed in cfg['seeds'] for method in cfg['methods']]

def run(job):
    seed,method=job
    out=ROOT/f'results/interaction-seed{seed}/{method}'
    if (out/'COMPLETE.json').exists():
        return
    with (ROOT/f'logs/interaction-{method}-seed{seed}.log').open('w') as log:
        process=subprocess.run([sys.executable,'-u',str(ROOT/'code/run_interaction.py'),
                                '--seed',str(seed),'--method',method],stdout=log,stderr=subprocess.STDOUT)
    print(seed,method,process.returncode,flush=True)
    if process.returncode:
        raise RuntimeError(job)

if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(run,jobs))
