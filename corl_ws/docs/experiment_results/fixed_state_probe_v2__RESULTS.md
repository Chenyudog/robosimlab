# Fixed-state displacement probes

Samples: {'train': 91, 'validation': 29, 'test': 31}; validation-selected: prefix_0; constant test RMSE: 5.531 mm.

| Feature | Validation MSE | Test coordinate RMSE(mm) | Test R2 | Correct direction | Shuffled labels RMSE(mm) |
|---|---:|---:|---:|---:|---:|
| prefix_0 | 0.00001452 | 4.388 | 0.371 | 0.806 | 5.702 |
| prefix_5 | 0.00001510 | 4.455 | 0.351 | 0.839 | 5.725 |
| prefix_11 | 0.00001516 | 4.432 | 0.358 | 0.839 | 5.725 |
| prefix_17 | 0.00001522 | 4.478 | 0.344 | 0.839 | 5.765 |
| expert_0_denoise0 | 0.00002801 | 5.415 | 0.041 | 0.581 | 5.440 |
| expert_5_denoise0 | 0.00002744 | 5.517 | 0.005 | 0.613 | 6.247 |
| expert_11_denoise0 | 0.00002930 | 5.571 | -0.014 | 0.516 | 6.200 |
| expert_17_denoise0 | 0.00002984 | 5.568 | -0.013 | 0.516 | 6.116 |
| expert_0_denoise9 | 0.00003352 | 5.919 | -0.145 | 0.419 | 5.786 |
| expert_5_denoise9 | 0.00003264 | 5.864 | -0.124 | 0.452 | 5.799 |
| expert_11_denoise9 | 0.00003223 | 5.752 | -0.081 | 0.419 | 5.749 |
| expert_17_denoise9 | 0.00003300 | 5.793 | -0.097 | 0.452 | 5.784 |
| pixel_mean | 0.00000102 | 2.416 | 0.809 | 1.000 | 5.599 |
| state_delta | 0.00003040 | 5.531 | 0.000 | 0.516 | 5.531 |

- One task, four held-out scenes; exploratory, no significance claim.
- Robot state, inference noise and task text fixed within each scene. Target labels never enter policy.
- Physically invalid shifts excluded individually; axes and directions may have unequal coverage.
- Readout of imposed image changes is not proof of action use, object semantics or closed-loop correction.
- Correct direction metric measures positive dot product, not precise angle accuracy.
