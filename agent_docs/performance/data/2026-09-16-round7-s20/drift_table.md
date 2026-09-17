| scene | n/arm | frames | meanFrameMs base → head (Δ%) | medianFrameMs | ms/newton | Newton tot | PCG tot | LS tot | disjoint(mean) | Welch p (mean) |
|---|---:|---:|---|---|---|---|---|---|---|---|
| crease-press | 10 | 130 | 269.68 → 233.80 (-13.30%) | 87.43 → 76.70 (-12.27%) | 62.70 → 54.23 (-13.51%) | 559 → 560 (+0.16%) | 115.2k → 118.6k (+2.99%) | 616 → 620 | no | 0.00011 |
| cube-wall-cloth | 5 | 100 | 56.71 → 56.28 (-0.76%) | 50.66 → 51.12 (+0.90%) | 11.08 → 10.77 (-2.79%) | 512 → 523 (+2.11%) | 19.8k → 20.2k (+1.74%) | 532 → 547 | no | 0.75 |
| tumbler-garments | 5 | 180 | 112.94 → 105.87 (-6.26%) | 104.09 → 97.47 (-6.35%) | 13.47 → 13.14 (-2.51%) | 1508 → 1452 (-3.75%) | 51.1k → 46.2k (-9.65%) | 1784 → 1725 | no | 0.052 |
| stiff-gipc-case2 | 5 | 250 | 155.42 → 153.90 (-0.98%) | 158.29 → 154.95 (-2.11%) | 23.38 → 23.17 (-0.91%) | 1662 → 1661 (-0.07%) | 64.4k → 63.6k (-1.19%) | 1786 → 1775 | no | 0.046 |
| mas-bunny | 5 | 100 | 61.46 → 61.01 (-0.73%) | 67.63 → 67.16 (-0.69%) | 13.22 → 13.12 (-0.73%) | 465 → 465 (+0.00%) | 35.2k → 35.2k (-0.05%) | 465 → 465 | yes | 0.035 |
| rigid-wrecking-balls | 5 | 120 | 25.46 → 25.09 (-1.46%) | 22.41 → 22.02 (-1.73%) | 6.51 → 6.46 (-0.89%) | 469 → 466 (-0.55%) | 12.8k → 12.7k (-0.77%) | 484 → 479 | no | 0.44 |

per-run mean ms/frame:
  crease-press         base  267.9 254.4 263.4 256.8 270.5 264.5 273.4 273.6 298.2 274.2   warm-up(discarded): 262.8
  crease-press         head  221.4 231.2 229.9 244.5 243.1 219.4 279.1 228.7 226.3 214.4   warm-up(discarded): 216.8
  cube-wall-cloth      base  56.1 55.3 56.5 58.2 57.5   warm-up(discarded): 58.1
  cube-wall-cloth      head  55.5 54.2 54.6 56.4 60.6   warm-up(discarded): 55.2
  tumbler-garments     base  109.3 107.7 117.3 119.0 111.4   warm-up(discarded): 110.9
  tumbler-garments     head  111.9 103.2 104.4 109.7 100.1   warm-up(discarded): 116.5
  stiff-gipc-case2     base  155.0 154.9 155.2 156.6 155.4   warm-up(discarded): 154.5
  stiff-gipc-case2     head  153.7 154.3 155.5 152.2 153.9   warm-up(discarded): 155.5
  mas-bunny            base  61.3 61.3 61.2 61.5 62.0   warm-up(discarded): 61.4
  mas-bunny            head  61.0 60.8 61.1 61.1 61.1   warm-up(discarded): 61.0
  rigid-wrecking-balls base  25.1 25.4 26.8 25.5 24.4   warm-up(discarded): 25.6
  rigid-wrecking-balls head  25.8 25.2 25.0 24.5 25.1   warm-up(discarded): 25.5
