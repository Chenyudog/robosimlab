# Activation probe pilot

Samples: {'train': 39, 'validation': 14, 'test': 17}. Validation-selected activation: expert_17_denoise0.

| Feature | Validation MSE | Test RMSE (m) | Test R2 vs train mean |
|---|---:|---:|---:|
| prefix_0 | 0.001507 | 0.0736 | 0.606 |
| prefix_5 | 0.001088 | 0.0835 | 0.493 |
| prefix_11 | 0.001290 | 0.0840 | 0.486 |
| prefix_17 | 0.001380 | 0.0858 | 0.465 |
| expert_0_denoise0 | 0.000246 | 0.0688 | 0.656 |
| expert_5_denoise0 | 0.000122 | 0.0531 | 0.795 |
| expert_11_denoise0 | 0.000105 | 0.0632 | 0.709 |
| expert_17_denoise0 | 0.000100 | 0.0629 | 0.712 |
| expert_0_denoise9 | 0.000678 | 0.0682 | 0.662 |
| expert_5_denoise9 | 0.000190 | 0.0619 | 0.721 |
| expert_11_denoise9 | 0.000317 | 0.0672 | 0.672 |
| expert_17_denoise9 | 0.000252 | 0.0650 | 0.693 |
| state | 0.000097 | 0.0343 | 0.915 |

- One task and four held-out trajectories; no significance claim.
- Layer selection by validation only; test is descriptive, not for subsequent tuning.
- Decodability is not causal use; state baseline may explain phase correlations.
- Pooled activations can miss spatial information; shuffled rows are a weak negative control.
