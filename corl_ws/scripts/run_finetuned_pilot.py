"""Autonomously wait for verified checkpoint, establish baseline, then paired pilot."""
from pathlib import Path
import json,sys,time,os
import torch
import run_baseline as baseline
from run_paired_rollouts import run_pairs
ROOT=baseline.ROOT
PREFIX=os.environ.get("VLA_RUN_PREFIX","finetuned")
REPLAN=int(os.environ.get("VLA_REPLAN_STEPS","10"))


def state(stage,**kw):
    baseline.save_json(ROOT/'outputs/pilot_status.json',{'stage':stage,'updated_unix':time.time(),**kw})
    print(stage,kw,flush=True)


def main():
    model=ROOT/'models/pi0_libero_finetuned'
    state('waiting_for_verified_finetuned_checkpoint')
    started=time.monotonic()
    while not (model/'verification.json').exists():
        if time.monotonic()-started>7200:raise TimeoutError('Verified checkpoint unavailable after two hours')
        time.sleep(10)
    source=json.loads((model/'verification.json').read_text())
    if not source['verified']:raise RuntimeError('Checkpoint verification failed')
    torch.set_num_threads(4)
    state('loading_finetuned_policy')
    policy=baseline.Policy(model,vision_fp32=True)
    baseline.Policy=lambda _,**kwargs:policy
    successes=[];reports=[]
    for initial in range(3):
        output=ROOT/f'outputs/{PREFIX}_spatial0_initial{initial}_h{REPLAN}_seed7'
        if not output.exists():
            state('running_ordinary_baseline',initial_state=initial)
            sys.argv=['run_baseline.py','--model-dir',str(model),'--output',str(output),
                '--task-id','0','--episode-index',str(initial),'--seed','7','--max-steps','220',
                '--replan-steps',str(REPLAN),'--snapshot-step','10','--vision-fp32']
            baseline.main()
        result=json.loads((output/'result.json').read_text())
        reports.append({'initial_state':initial,'success':result['success'],'output':str(output)})
        if result['success']:successes.append(output)
    baseline.save_json(ROOT/f'outputs/{PREFIX}_baseline_summary.json',{'episodes':reports,'note':'Three exploratory cases, not benchmark accuracy.'})
    if len(successes)<2:
        state('baseline_gate_not_met',successful_episodes=len(successes),total=3,
              next_action='Review official inference compatibility before any feedback claims or SAE training.')
        return
    state('running_paired_perturbation_pilot',baseline=str(successes[0]))
    summary=run_pairs(policy,successes[0],ROOT/f'outputs/{PREFIX}_paired_pilot',max_steps=200)
    state('paired_pilot_complete',branches=summary,
          next_action='Check nominal paired branch and perturbation validity before interpreting response differences.')

if __name__=='__main__':
    try:main()
    except BaseException as e:
        state('error',error_type=type(e).__name__,message=str(e));raise
