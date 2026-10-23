"""Three normal-control diagnostic episodes; reuse one loaded policy."""
import json
from pathlib import Path
import sys
import time
import torch
import run_baseline as baseline

ROOT=baseline.ROOT


def main():
    # Avoid competing for VRAM with the separately launched offline audit.
    audit=ROOT/'outputs/offline_inference_audit.json'
    if not audit.exists():raise RuntimeError('Finish offline audit before starting this batch')
    torch.set_num_threads(4)
    policy=baseline.Policy(ROOT/'models/pi0_libero')
    baseline.Policy=lambda _,**kwargs:policy
    reports=[]
    for task,episode in [(0,0),(0,1),(1,0)]:
        out=ROOT/f'outputs/diagnostic_spatial{task}_initial{episode}_h10_seed7'
        sys.argv=['run_baseline.py','--output',str(out),'--task-id',str(task),
                  '--episode-index',str(episode),'--seed','7','--max-steps','220','--replan-steps','10']
        print(f'RUN {out.name}',flush=True)
        try:
            baseline.main()
            result=json.loads((out/'result.json').read_text())
            reports.append({'task_id':task,'initial_state_index':episode,'success':result['success'],'steps':result['steps'],'output':str(out)})
        except Exception as exc:
            reports.append({'task_id':task,'initial_state_index':episode,'error':str(exc),'output':str(out)})
            baseline.save_json(ROOT/'outputs/diagnostic_batch.json',{'episodes':reports,'status':'incomplete'})
            raise
        baseline.save_json(ROOT/'outputs/diagnostic_batch.json',{'episodes':reports,'status':'running'})
    baseline.save_json(ROOT/'outputs/diagnostic_batch.json',{'episodes':reports,'status':'complete',
        'note':'Small diagnostic sample across two tasks; not a benchmark success-rate estimate.'})
    print('DIAGNOSTIC BATCH COMPLETE',flush=True)

if __name__=='__main__':main()
