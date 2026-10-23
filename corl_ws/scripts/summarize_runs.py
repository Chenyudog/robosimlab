"""Generate an evidence ledger from actual evaluation artifacts."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
rows=[]
for p in sorted((ROOT/'outputs').glob('*/manifest.json')):
    manifest=json.loads(p.read_text())
    if 'baseline_output' in manifest:continue
    result=p.parent/'result.json';error=p.parent/'error.json'
    if result.exists():
        d=json.loads(result.read_text());status='success' if d.get('success') else 'task not completed';steps=d.get('steps','')
    elif error.exists():
        status='runtime interrupted/error';steps=''
    else:
        status='no terminal result yet';steps=''
    rows.append(f"| {p.parent.name} | {manifest.get('task_id','')} | {manifest.get('episode_index','')} | {manifest.get('replan_steps','')} | {manifest.get('vision_fp32',False)} | {status} | {steps} |")
text='''# Experiment status

Generated from saved manifests and terminal results. Runtime errors are separate from task failures. This small diagnostic set is not a benchmark success-rate estimate and does not establish feedback failure.

| Run | Task ID | Initial state | Replan interval | FP32 vision | Outcome | Steps |
|---|---:|---:|---:|---|---|---:|
'''+ '\n'.join(rows)+'''\n
## Verified infrastructure

- LIBERO model file downloaded and SHA256 verified (`models/pi0_libero/verification.json`).
- Environment-only paired displacement and initial-state replay check passed (`outputs/environment_fork_check_run2/result.json`).
- Low-memory checkpoint loader reads tensors in bounded host-memory chunks.
- Six checkpoint normalization buffers are consumed; state/action normalization uses the checkpoint's own saved statistics.

## Research gate

Establish a task-capable ordinary-control baseline before interpreting perturbation failures or training an SAE repair. No feedback repair training has been started.
'''
text+='\n## Paired pilots\n\n'
for p in sorted((ROOT/'outputs').glob('*/summary.json')):
    d=json.loads(p.read_text())
    if 'branches' not in d:continue
    text+=f'- {p.parent.name}: {d.get("status", "unknown")}\n'
    for b in d['branches']:
        text+=f'  - {b["branch"]}: {b["status"]}; success={b.get("success", "not applicable")}\n'
(ROOT/'EXPERIMENT_STATUS.md').write_text(text)
print(text)
