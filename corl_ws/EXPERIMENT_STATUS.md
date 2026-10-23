# Experiment status

Generated from saved manifests and terminal results. Runtime errors are separate from task failures. This small diagnostic set is not a benchmark success-rate estimate and does not establish feedback failure.

| Run | Task ID | Initial state | Replan interval | FP32 vision | Outcome | Steps |
|---|---:|---:|---:|---|---|---:|
| baseline_spatial0_openpi_delta_fp32 | 0 | 0 | 5 | True | task not completed | 220 |
| baseline_spatial0_seed7 | 0 | 0 | 5 | False | runtime interrupted/error |  |
| baseline_spatial0_seed7_run2 | 0 | 0 | 5 | False | runtime interrupted/error |  |
| baseline_spatial0_seed7_run3 | 0 | 0 | 5 | False | runtime interrupted/error |  |
| baseline_spatial0_seed7_run4 | 0 | 0 | 5 | False | task not completed | 220 |
| baseline_spatial0_seed7_vision_fp32 | 0 | 0 | 10 | True | task not completed | 220 |
| diagnostic_spatial0_initial0_h10_seed7 | 0 | 0 | 10 | False | task not completed | 220 |
| diagnostic_spatial0_initial1_h10_seed7 | 0 | 1 | 10 | False | task not completed | 220 |
| diagnostic_spatial1_initial0_h10_seed7 | 1 | 0 | 10 | False | task not completed | 220 |
| finetuned_spatial0_initial0_h10_seed7 | 0 | 0 | 10 | True | task not completed | 220 |
| finetuned_spatial0_initial1_h10_seed7 | 0 | 1 | 10 | True | task not completed | 220 |
| finetuned_spatial0_initial2_h10_seed7 | 0 | 2 | 10 | True | task not completed | 220 |
| legacy_aligned_spatial0_initial0_h10_seed7 | 0 | 0 | 10 | True | task not completed | 220 |
| legacy_aligned_spatial0_initial1_h10_seed7 | 0 | 1 | 10 | True | success | 156 |
| legacy_aligned_spatial0_initial2_h10_seed7 | 0 | 2 | 10 | True | task not completed | 220 |
| legacy_finetuned_spatial0_initial0_h10_seed7 | 0 | 0 | 10 | True | no terminal result yet |  |
| legacy_h50_spatial0_initial0_h50_seed7 | 0 | 0 | 50 | True | success | 77 |
| legacy_h50_spatial0_initial1_h50_seed7 | 0 | 1 | 50 | True | success | 79 |
| legacy_h50_spatial0_initial2_h50_seed7 | 0 | 2 | 50 | True | task not completed | 220 |

## Verified infrastructure

- LIBERO model file downloaded and SHA256 verified (`models/pi0_libero/verification.json`).
- Environment-only paired displacement and initial-state replay check passed (`outputs/environment_fork_check_run2/result.json`).
- Low-memory checkpoint loader reads tensors in bounded host-memory chunks.
- Six checkpoint normalization buffers are consumed; state/action normalization uses the checkpoint's own saved statistics.

## Research gate

Establish a task-capable ordinary-control baseline before interpreting perturbation failures or training an SAE repair. No feedback repair training has been started.

## Paired pilots

- legacy_h50_paired_5mm: complete
  - nominal: complete; success=True
  - minus_x: complete; success=False
  - plus_x: complete; success=False
- legacy_h50_paired_5mm_initial1: complete
  - nominal: complete; success=True
  - minus_x: complete; success=False
  - plus_x: complete; success=True
- legacy_h50_paired_pilot: complete
  - nominal: complete; success=False
  - minus_x: complete; success=False
  - plus_x: invalid_initial_interpenetration; success=not applicable
