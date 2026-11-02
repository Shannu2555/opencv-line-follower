# Results (216 runs: 3 tracks x 2 lighting x 3 speeds x 4 controllers x 3 seeds)

Error stats are over completed laps only. Latency = per-frame perception time on one CPU core of my dev machine.

| controller   |   runs |   completed | completion   | clean   | hard   |   cte_rms_cm |   cte_max_cm |   head_rms_deg |   lat_med_ms |   lat_p95_ms |
|:-------------|-------:|------------:|:-------------|:--------|:-------|-------------:|-------------:|---------------:|-------------:|-------------:|
| Centroid PID |     54 |          27 | 50%          | 24/27   | 3/27   |         0.88 |         11.2 |            1.7 |         0.09 |         0.13 |
| Fit + PID    |     54 |          52 | 96%          | 27/27   | 25/27  |         2.64 |         10.5 |            2.5 |         1.93 |         2.21 |
| Pure pursuit |     54 |          51 | 94%          | 27/27   | 24/27  |         0.86 |         12   |            1.9 |         1.92 |         2.18 |
| Stanley + FF |     54 |          49 | 91%          | 25/27   | 24/27  |         0.48 |          8.8 |            1   |         1.93 |         2.18 |

## Laps completed out of 6 per cell (clean + hard, 3 seeds each)

|                        |   Centroid PID |   Fit + PID |   Pure pursuit |   Stanley + FF |
|:-----------------------|---------------:|------------:|---------------:|---------------:|
| ('figure_eight', 0.12) |              0 |           6 |              6 |              6 |
| ('figure_eight', 0.19) |              3 |           6 |              6 |              6 |
| ('figure_eight', 0.26) |              3 |           6 |              6 |              6 |
| ('oval_chicane', 0.12) |              3 |           6 |              6 |              6 |
| ('oval_chicane', 0.19) |              6 |           6 |              6 |              6 |
| ('oval_chicane', 0.26) |              3 |           6 |              6 |              6 |
| ('technical', 0.12)    |              3 |           6 |              5 |              1 |
| ('technical', 0.19)    |              3 |           5 |              5 |              6 |
| ('technical', 0.26)    |              3 |           5 |              5 |              6 |

## Like-for-like: 8 track/light/speed cases that all 4 finished on every seed

| controller   |   cte_rms_cm |   cte_max_cm |   head_rms_deg |   lap_time |
|:-------------|-------------:|-------------:|---------------:|-----------:|
| Centroid PID |         0.78 |         2.86 |           1.6  |      63.69 |
| Fit + PID    |         2.27 |         5.26 |           2.08 |      62.53 |
| Pure pursuit |         0.84 |         3.33 |           1.82 |      63.82 |
| Stanley + FF |         0.47 |         1.58 |           0.97 |      64.19 |
