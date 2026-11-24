# Fairness report (test split)

Cutoff 0.160; n = 6,000. Positive = predicted default = decline. Brackets are stratified bootstrap 95% intervals.

## SEX

Approval-rate ratio (lowest / highest group): 0.944; FPR gap 0.022; FNR gap 0.022; observed default rate 0.213 to 0.234.

| Group | n | Default rate | Decline rate | Precision | Recall | FPR | FNR | Mean predicted | Calibration gap |
|---|---|---|---|---|---|---|---|---|---|
| female | 3,598 | 0.213 | 0.471 [0.455, 0.486] | 0.347 | 0.768 [0.738, 0.799] | 0.391 [0.373, 0.408] | 0.232 [0.201, 0.262] | 0.219 | 0.006 [0.000, 0.011] |
| male | 2,402 | 0.234 | 0.501 [0.481, 0.522] | 0.368 | 0.790 [0.759, 0.824] | 0.413 [0.392, 0.437] | 0.210 [0.176, 0.241] | 0.228 | -0.006 [-0.012, 0.001] |

## EDUCATION

Approval-rate ratio (lowest / highest group): 0.744; FPR gap 0.155; FNR gap 0.059; observed default rate 0.190 to 0.257.

| Group | n | Default rate | Decline rate | Precision | Recall | FPR | FNR | Mean predicted | Calibration gap |
|---|---|---|---|---|---|---|---|---|---|
| graduate school | 2,130 | 0.190 | 0.404 [0.383, 0.423] | 0.350 | 0.743 [0.697, 0.780] | 0.324 [0.303, 0.347] | 0.257 [0.220, 0.303] | 0.192 | 0.002 [-0.005, 0.008] |
| high school | 1,014 | 0.257 | 0.556 [0.528, 0.584] | 0.360 | 0.778 [0.724, 0.826] | 0.479 [0.448, 0.517] | 0.222 [0.174, 0.276] | 0.246 | -0.011 [-0.022, 0.000] |
| others / unlabelled (0, 4, 5, 6) (small) | 82 | 0.061 | 0.073 [0.024, 0.134] | 0.167 | 0.200 [0.000, 0.600] | 0.065 [0.026, 0.130] | 0.800 [0.400, 1.000] | 0.102 | 0.041 [0.020, 0.067] |
| university | 2,774 | 0.236 | 0.529 [0.509, 0.546] | 0.358 | 0.802 [0.772, 0.833] | 0.445 [0.422, 0.466] | 0.198 [0.167, 0.228] | 0.241 | 0.004 [-0.003, 0.010] |

## MARRIAGE

Approval-rate ratio (lowest / highest group): 0.949; FPR gap 0.048; FNR gap 0.017; observed default rate 0.205 to 0.241.

| Group | n | Default rate | Decline rate | Precision | Recall | FPR | FNR | Mean predicted | Calibration gap |
|---|---|---|---|---|---|---|---|---|---|
| married | 2,767 | 0.241 | 0.468 [0.448, 0.485] | 0.396 | 0.768 [0.735, 0.801] | 0.373 [0.351, 0.392] | 0.232 [0.199, 0.265] | 0.222 | -0.020 [-0.026, -0.013] |
| others / unlabelled (0, 3) (small) | 75 | 0.160 | 0.533 [0.420, 0.627] | 0.250 | 0.833 [0.667, 1.000] | 0.476 [0.341, 0.587] | 0.167 [0.000, 0.333] | 0.225 | 0.065 [0.030, 0.111] |
| single | 3,158 | 0.205 | 0.495 [0.480, 0.514] | 0.325 | 0.785 [0.755, 0.819] | 0.420 [0.401, 0.440] | 0.215 [0.181, 0.245] | 0.223 | 0.018 [0.012, 0.023] |

## AGE_BAND

Approval-rate ratio (lowest / highest group): 0.863; FPR gap 0.075; FNR gap 0.051; observed default rate 0.206 to 0.237.

| Group | n | Default rate | Decline rate | Precision | Recall | FPR | FNR | Mean predicted | Calibration gap |
|---|---|---|---|---|---|---|---|---|---|
| 21-29 | 1,912 | 0.223 | 0.523 [0.504, 0.544] | 0.343 | 0.803 [0.760, 0.843] | 0.442 [0.417, 0.465] | 0.197 [0.157, 0.240] | 0.239 | 0.015 [0.008, 0.023] |
| 30-39 | 2,198 | 0.206 | 0.447 [0.429, 0.465] | 0.347 | 0.753 [0.709, 0.792] | 0.368 [0.348, 0.391] | 0.247 [0.208, 0.291] | 0.206 | 0.000 [-0.005, 0.008] |
| 40-49 | 1,341 | 0.233 | 0.472 [0.449, 0.497] | 0.381 | 0.772 [0.729, 0.814] | 0.381 [0.355, 0.409] | 0.228 [0.186, 0.271] | 0.219 | -0.013 [-0.022, -0.004] |
| 50-59 | 473 | 0.237 | 0.514 [0.469, 0.556] | 0.362 | 0.786 [0.701, 0.862] | 0.429 [0.384, 0.474] | 0.214 [0.138, 0.299] | 0.235 | -0.002 [-0.018, 0.015] |
| 60+ (small) | 76 | 0.303 | 0.513 [0.421, 0.618] | 0.462 | 0.783 [0.609, 0.957] | 0.396 [0.283, 0.528] | 0.217 [0.043, 0.391] | 0.242 | -0.060 [-0.095, -0.023] |

## Limitations

- Groups use attributes recorded in the data; nothing is inferred.
- SEX, MARRIAGE and AGE are not model inputs, but other features can act as proxies, so group outcomes can still differ.
- Equal FPR/FNR (equalised odds), equal decline rates (demographic parity) and equal calibration cannot all hold when default rates differ between groups; the report shows all three instead of declaring one 'fair'.
- Passing any of these metrics is not evidence of legal or regulatory compliance.
- Groups under 100 applicants have wide intervals and are flagged.
