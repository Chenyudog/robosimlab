# VLA feedback research project

This directory is independent of the existing robot scene/control project in the parent directory.

- `scripts/`: new closed-loop evaluation code.
- `models/pi0_libero/`: pinned LIBERO-adapted pi0, tokenizer and normalization statistics.
- `data/libero_pi/`: downloaded demonstration dataset.
- `runtime/`: the existing LIBERO Python environment and environment smoke tests.
- `reference/pi0_representation/`: earlier pretrained-feature extraction scripts/results; these are not feedback research results.
- `docs/`: research proposal.
- `outputs/`: new evaluation videos, trajectories and JSON results.
- `logs/`: process logs.
- `migration.json`: migration record.

Compatibility symlinks at the former paths keep existing commands and the relocated virtual environment usable. Do not remove these links until the virtual environment and all legacy scripts have been rebuilt with new paths.

The shared LeRobot source dependency remains `/home/cyd/lerobot/src`; existing base models remain in the parent's `model/`. Neither is a new research output. No parent README or robot scene was changed.

## Run a single normal baseline episode

```bash
cd /home/cyd/robosimlab/vla_feedback
./run_baseline.sh --output outputs/baseline_spatial0_seed7 --suite libero_spatial --task-id 0 --seed 7 --max-steps 220 --replan-steps 5
```

Use a fresh output path for every run. The script rotates raw LIBERO camera images by 180 degrees, uses mean/std normalization for pi0 (not pi0.5 quantile normalization), and restores 7D LIBERO actions before execution. It strictly checks checkpoint parameters. This is simulation only.

An environment smoke test does not establish policy success. A completed single policy episode is not a benchmark success-rate estimate. Inspect `result.json` and `rollout.mp4`; exceptions are recorded as `error.json` and are not counted as task failures.

The local LeRobot port, BF16 precision and image resizing are recorded in each manifest; no exact match to published OpenPI results is claimed. No feedback repair has been trained yet.

## Counterfactual environment check

`outputs/environment_fork_check_run2/result.json` records a passing environment-only check: target translations of +/-2 cm, identical restored images, and repeated five-step physics state discrepancy below 1e-9. This is one initial snapshot and does not establish policy feedback or universal replay correctness.

The baseline runner saves an internal simulator/controller snapshot at policy step 40 when reached. It is intended for subsequent perturbation experiments and does not prove task success.
