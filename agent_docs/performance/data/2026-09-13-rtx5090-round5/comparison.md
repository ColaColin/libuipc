GPU: NVIDIA GeForce RTX 5090, 595.71.05
| benchmark | frames | side | commit | n | mean ms/frame (mean of runs) | median-of-medians | min–max mean | Newton/frame | PCG/frame |
|---|---:|---|---|---:|---:|---:|---|---:|---:|
| cube-wall-cloth (100 frames) | 100 | base | 890482c2 | 5 | 46.61 | 41.14 | 45.8–47.3 | 5.08 | 198.3 |
| cube-wall-cloth (100 frames) | 100 | head | 31ff6a05 | 5 | 34.18 | 30.49 | 33.6–35.0 | 5.10 | 198.9 |
| cube-wall-cloth (100 frames) | | **Δ head vs base** | | | **-26.7 %** | **-25.9 %** | | +0.01 | +0.6 |
| mas-bunny (3 frames) | 3 | base | 890482c2 | 1 | 15.03 | 7.89 | 15.0–15.0 | 2.00 | 50.0 |
| mas-bunny (3 frames) | 3 | head | 31ff6a05 | 1 | 11.00 | 6.55 | 11.0–11.0 | 2.00 | 50.0 |
| mas-bunny (3 frames) | | **Δ head vs base** | | | **-26.8 %** | **-16.9 %** | | +0.00 | +0.0 |
| mas-bunny (100 frames) | 100 | base | 890482c2 | 5 | 25.88 | 28.56 | 25.8–25.9 | 4.65 | 352.2 |
| mas-bunny (100 frames) | 100 | head | 31ff6a05 | 5 | 24.18 | 26.65 | 24.1–24.2 | 4.65 | 352.1 |
| mas-bunny (100 frames) | | **Δ head vs base** | | | **-6.6 %** | **-6.7 %** | | +0.00 | -0.1 |
| rigid-wrecking-balls (120 frames) | 120 | base | 890482c2 | 5 | 29.31 | 24.40 | 28.8–29.7 | 3.96 | 108.1 |
| rigid-wrecking-balls (120 frames) | 120 | head | 31ff6a05 | 5 | 20.51 | 17.23 | 19.9–20.8 | 4.00 | 108.3 |
| rigid-wrecking-balls (120 frames) | | **Δ head vs base** | | | **-30.1 %** | **-29.4 %** | | +0.04 | +0.2 |
| stiff-gipc-case2 (250 frames) | 250 | base | 890482c2 | 5 | 76.80 | 79.96 | 76.6–77.0 | 6.64 | 257.4 |
| stiff-gipc-case2 (250 frames) | 250 | head | 31ff6a05 | 5 | 57.28 | 58.97 | 56.9–57.7 | 6.64 | 258.5 |
| stiff-gipc-case2 (250 frames) | | **Δ head vs base** | | | **-25.4 %** | **-26.2 %** | | -0.00 | +1.1 |
