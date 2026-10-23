"""Fixed task0 initial states 0..19, followed by all successful-snapshot pairs."""
import json,sys,time,traceback
import torch
import run_baseline as baseline
from run_paired_rollouts import run_pairs
ROOT=baseline.ROOT
OUT=ROOT/'outputs/expanded_h50'

def status(stage,**kwargs):
    baseline.save_json(OUT/'status.json',dict(stage=stage,updated_unix=time.time(),**kwargs))
    print(stage,kwargs,flush=True)

def main():
    OUT.mkdir(exist_ok=True)
    protocol={'suite':'libero_spatial','task_id':0,'initial_states':list(range(20)),
        'seed':7,'replan_steps':50,'max_steps':220,'snapshot_step':10,
        'paired_selection':'All successful baselines, no selection by perturbation outcome',
        'paired_delta_m':.005,'paired_replan_steps':50,'paired_max_steps':220,
        'scope':'Single-task diagnostic. Not LIBERO benchmark accuracy. Existing initial0..2 reused.',
        'limitation':'All branches start a new chunk at snapshot10. Success is limited to 220 steps.'}
    protocol_file=OUT/'protocol.json'
    if protocol_file.exists():
        if json.loads(protocol_file.read_text())!=protocol:raise RuntimeError('Protocol drift')
    else:baseline.save_json(protocol_file,protocol)
    torch.set_num_threads(4)
    status('loading_model')
    model=ROOT/'models/pi0_libero_finetuned'
    policy=baseline.Policy(model,vision_fp32=True)
    baseline.Policy=lambda _,**kwargs:policy
    reports=[]
    for initial in range(20):
        output=(ROOT/f'outputs/legacy_h50_spatial0_initial{initial}_h50_seed7' if initial<3
                else OUT/f'baseline_initial{initial}')
        if not (output/'result.json').exists():
            if output.exists():raise RuntimeError(f'Incomplete run requires inspection: {output}')
            status('baseline',initial=initial)
            sys.argv=['run_baseline.py','--model-dir',str(model),'--output',str(output),
                '--task-id','0','--episode-index',str(initial),'--seed','7','--max-steps','220',
                '--replan-steps','50','--snapshot-step','10','--vision-fp32']
            baseline.main()
        meta=json.loads((output/'manifest.json').read_text())
        for key,value in [('replan_steps',50),('episode_index',initial),('seed',7),('max_steps',220),('task_id',0),('backend','legacy_pi0')]:
            if meta[key]!=value:raise RuntimeError(f'Reused run mismatch {key}: {output}')
        r=json.loads((output/'result.json').read_text())
        reports.append(dict(initial_state=initial,success=r['success'],steps=r['steps'],output=str(output)))
        baseline.save_json(OUT/'baseline_summary.json',{'episodes':reports,'completed':len(reports),'successes':sum(x['success'] for x in reports)})
    pairs=[]
    for r in reports:
        if not r['success']:continue
        i=r['initial_state']
        output=(ROOT/'outputs'/('legacy_h50_paired_5mm' if i==0 else 'legacy_h50_paired_5mm_initial1') if i<2
                else OUT/f'paired_initial{i}')
        status('paired',initial=i)
        if (output/'validation_error.json').exists():
            error=json.loads((output/'validation_error.json').read_text())
            branches=[dict(branch='snapshot',status='invalid_replay',reason=error['error'])]
        elif not (output/'summary.json').exists():
            try:
                branches=run_pairs(policy,r['output'],output,max_steps=220,delta=.005,replan_steps=50)
            except RuntimeError as exc:
                if not str(exc).startswith('Mid-episode replay not reproducible:'):raise
                baseline.save_json(output/'validation_error.json',{'error':str(exc),'count_as_policy_failure':False})
                branches=[dict(branch='snapshot',status='invalid_replay',reason=str(exc))]
        else:
            d=json.loads((output/'summary.json').read_text())
            if d['status']!='complete':raise RuntimeError(f'Incomplete paired run: {output}')
            branches=d['branches']
        pairs.append(dict(initial_state=i,output=str(output),branches=branches))
        baseline.save_json(OUT/'paired_summary.json',{'episodes':pairs,'scope':'Conditional on ordinary baseline success; compare shifted branches with their matched nominal.'})
    status('complete',baseline_successes=sum(r['success'] for r in reports),baseline_total=20,paired_snapshots=len(pairs))

if __name__=='__main__':
    try:main()
    except BaseException as e:
        status('error',error=str(e),traceback=traceback.format_exc());raise
