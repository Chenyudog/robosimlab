# Spatial specificity v3

Target samples {'train': 91, 'validation': 29, 'test': 73}; distractor test 79; validation-selected prefix_17_grid4. Constant RMSE 5.504mm.

| Feature | Target test RMSE(mm) | R2 | Shuffled RMSE(mm) | Matched target norm(mm) | Matched distractor norm(mm) |
|---|---:|---:|---:|---:|---:|
| prefix_0_grid4 | 4.057 | 0.457 | 6.039 | 6.444 | 2.786 |
| prefix_5_grid4 | 3.812 | 0.520 | 5.686 | 6.119 | 2.492 |
| prefix_11_grid4 | 3.745 | 0.537 | 5.556 | 5.977 | 2.438 |
| prefix_17_grid4 | 3.708 | 0.546 | 5.525 | 5.891 | 2.446 |
| expert_0_denoise0_positions | 5.408 | 0.034 | 5.623 | 2.011 | 4.091 |
| expert_5_denoise0_positions | 5.565 | -0.022 | 5.725 | 1.751 | 1.200 |
| expert_11_denoise0_positions | 5.221 | 0.100 | 5.664 | 2.469 | 1.152 |
| expert_17_denoise0_positions | 5.221 | 0.100 | 5.695 | 2.258 | 1.223 |
| expert_0_denoise9_positions | 6.846 | -0.547 | 8.045 | 4.326 | 3.296 |
| expert_5_denoise9_positions | 7.031 | -0.632 | 7.160 | 4.480 | 3.101 |
| expert_11_denoise9_positions | 7.053 | -0.642 | 6.814 | 3.872 | 2.525 |
| expert_17_denoise9_positions | 6.723 | -0.492 | 6.861 | 3.474 | 2.321 |
| pixel_mean | 2.222 | 0.837 | 5.440 | 5.679 | 1.689 |
| state_delta | 5.504 | 0.000 | 5.504 | 0.275 | 0.275 |

- Single task; spatial pooling still loses information.
- A displacement readout is not evidence of causal use or semantic features.
- Distractor differs in spatial position; lower readout may reflect a spatial shortcut, not language goal binding.
- Target/distractor matching excludes invalid shifts; no statistical significance claim or closed-loop success measurement.
