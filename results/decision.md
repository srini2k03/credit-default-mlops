# Decision analysis (cost-sensitive cutoff)

> Educational decision-support analysis on a public 2005 dataset. Costs are stated assumptions in relative units; expected costs are not financial savings, and this is not an approved lending or credit-scoring system.

Policy: FN cost 5, FP cost 1 (ratio 5:1), Bayes cutoff 0.167; deployed cutoff 0.160. All rows: test split, cutoffs chosen on validation.

| Rule | Cutoff | Cost / applicant | Recall | Precision | Approval rate | FPR | FNR | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|
| deployed (cost-optimal on validation) | 0.160 | 0.5578 | 0.777 | 0.356 | 0.517 | 0.400 | 0.223 | 1031 | 1867 | 2806 | 296 |
| cost-optimal on validation (re-derived) | 0.160 | 0.5578 | 0.777 | 0.356 | 0.517 | 0.400 | 0.223 | 1031 | 1867 | 2806 | 296 |
| Bayes cutoff fp/(fp+fn) on calibrated scores | 0.167 | 0.5530 | 0.763 | 0.367 | 0.540 | 0.374 | 0.237 | 1013 | 1748 | 2925 | 314 |
| 0.5 (default classifier cutoff) | 0.500 | 0.7523 | 0.354 | 0.672 | 0.883 | 0.049 | 0.646 | 470 | 229 | 4444 | 857 |
| max F1 on validation | 0.334 | 0.6292 | 0.512 | 0.559 | 0.798 | 0.114 | 0.488 | 679 | 535 | 4138 | 648 |
| Youden J on validation | 0.225 | 0.5747 | 0.640 | 0.445 | 0.682 | 0.226 | 0.360 | 849 | 1058 | 3615 | 478 |

## Sensitivity to the cost ratio

| FN:FP | Bayes cutoff | Optimal cutoff (validation) | Test cost at optimum | Test cost at deployed cutoff | Regret of deployed cutoff | Test cost at 0.5 | Recall at optimum | Approval at optimum |
|---|---|---|---|---|---|---|---|---|
| 1:1 | 0.500 | 0.395 | 0.1837 | 0.3605 | +0.1768 | 0.1810 | 0.439 | 0.843 |
| 1.5:1 | 0.400 | 0.395 | 0.2457 | 0.3852 | +0.1395 | 0.2524 | 0.439 | 0.843 |
| 2:1 | 0.333 | 0.335 | 0.3053 | 0.4098 | +0.1045 | 0.3238 | 0.510 | 0.798 |
| 3:1 | 0.250 | 0.225 | 0.4153 | 0.4592 | +0.0438 | 0.4667 | 0.640 | 0.682 |
| 4:1 | 0.200 | 0.225 | 0.4950 | 0.5085 | +0.0135 | 0.6095 | 0.640 | 0.682 |
| 5:1 | 0.167 | 0.160 | 0.5578 | 0.5578 | +0.0000 | 0.7523 | 0.777 | 0.517 |
| 6:1 | 0.143 | 0.150 | 0.6023 | 0.6072 | +0.0048 | 0.8952 | 0.804 | 0.480 |
| 8:1 | 0.111 | 0.125 | 0.6635 | 0.7058 | +0.0423 | 1.1808 | 0.861 | 0.391 |
| 10:1 | 0.091 | 0.080 | 0.6975 | 0.8045 | +0.1070 | 1.4665 | 0.941 | 0.224 |
| 12:1 | 0.077 | 0.070 | 0.7202 | 0.9032 | +0.1830 | 1.7522 | 0.957 | 0.182 |
| 15:1 | 0.062 | 0.060 | 0.7493 | 1.0512 | +0.3018 | 2.1807 | 0.968 | 0.141 |
| 20:1 | 0.048 | 0.040 | 0.7567 | 1.2978 | +0.5412 | 2.8948 | 0.993 | 0.054 |
