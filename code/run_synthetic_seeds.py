from pathlib import Path
import subprocess,sys,concurrent.futures
root=Path(__file__).resolve().parents[1]
(root/"logs").mkdir(exist_ok=True)
def run(seed):
    with (root/'logs'/f'synthetic-seed{seed}.log').open('w') as log:
        p=subprocess.run([sys.executable,'-u',str(root/'code/run_continual.py'),'--config',str(root/'configs/synthetic.json'),'--cache',str(root/'data/lowaltitude/cache'),'--output',str(root/f'results/synthetic-seed{seed}'),'--seed',str(seed)],stdout=log,stderr=subprocess.STDOUT)
    print(seed,p.returncode,flush=True)
    if p.returncode: raise RuntimeError(seed)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(run,[7,11,23,47,59,71,89,107]))
