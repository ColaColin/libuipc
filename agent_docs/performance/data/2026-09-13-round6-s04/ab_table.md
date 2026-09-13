| scene | `filter_toi` share of GPU kernel time | predicted | `meanFrameMs` | `ms_per_newton` | Newton / PCG |
|---|---:|---:|---|---|---|
| **mas-bunny** | 1.30 % | -0.33 % | -0.38 % (**disjoint**, p=0.0027) | -0.38 % (**disjoint**, p=0.0027) | +0.00 % / -0.00 % |
| **cube-wall-cloth** | 5.83 % | -1.46 % | +0.64 % (overlapping, p=0.77) | -1.49 % (**disjoint**, p=0.012) | +2.16 % / +1.07 % |
| **rigid-wrecking-balls** | 5.78 % | -1.45 % | -0.95 % (overlapping, p=0.41) | -1.79 % (overlapping, p=0.038) | +0.85 % / +0.72 % |
| **stiff-gipc-case2** | 6.25 % | -1.56 % | -1.48 % (**disjoint**, p=8.2e-05) | -1.54 % (**disjoint**, p=0.00052) | +0.06 % / -0.04 % |
| **tumbler-garments** | 13.42 % | -3.35 % | -3.86 % (overlapping, p=0.43) | -1.40 % (overlapping, p=0.75) | -2.43 % / -1.86 % |
