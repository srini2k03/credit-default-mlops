# Interview notes

Short answers to the questions this project invites, with the evidence for each.

**Why is the cutoff not 0.5?** Because a missed default costs about five times a false alarm
(an argued assumption). The cost-optimal cutoff on validation is 0.16 (Bayes: 0.167); on test it
lowers expected cost per applicant from 0.752 to 0.558 and raises default recall from 35% to 78%,
at the price of approving 52% instead of 88% of applicants. Whether that trade is acceptable is a
business decision; the sensitivity table shows what each other ratio implies.

**Why select on cost and not ROC-AUC?** ROC-AUC measures ranking across all cutoffs; the
business only uses one. The four tree families are within each other's bootstrap intervals on
ROC-AUC and PR-AUC; validation cost decides, PR-AUC breaks ties, and paired differences are
published so nobody over-reads a 0.002 gap.

**Why calibrate, and why Platt here?** Class weights inflate scores (uncalibrated ECE 0.19); a
cost rule needs real probabilities. Platt and isotonic both fix it (ECE 0.015 / 0.019), but
isotonic's step function creates ties that cost 0.026 PR-AUC. The method is chosen per family on
a split neither calibrator was fitted on.

**How do you know there is no leakage?** Splits are disjoint by ID; feature engineering is
row-wise (tested); scaler, encoder and SMOTE are fitted inside the pipeline on training rows; no
single feature has univariate ROC-AUC above 0.69; 0.18% of test rows share a feature vector with
a training row. The one thing that cannot be checked is time: there is no application date, so
metrics are in-period.

**What stops a bad model from shipping?** A 12-check promotion policy against the current
champion on the same test split: performance floors and regression limits, calibration (ECE,
Brier), expected cost, and operational checks (API contract, valid deterministic output, latency,
size). In the simulation an uncalibrated candidate with *lower* expected cost is rejected on
calibration, and a candidate trained on a corrupted outcome feed is rejected on PR-AUC, Brier and
cost. Failed challengers are recorded, not hidden.

**How do you roll back?** Promotion keeps `@previous-production`; rollback swaps the aliases. For
the Render deployment, which serves the committed bundle, rollback restores `models/bundled/`
from the previous bundle commit and the push redeploys. Both are tested.

**When do you retrain?** Not on a schedule and not on significance alone. Broken feeds (missing
values, unseen codes) are blocked, because retraining on them makes things worse. Without labels,
large feature or score drift starts a gated challenger. With labels, the decision follows
prevalence-robust performance (ROC-AUC drop, calibration gap, cost regret). The concept-drift
scenario shows why labels matter: nothing in the inputs moves, yet the deployed cutoff costs 22%
more than the optimum.

**Is the model fair?** Sex, marital status and age are not inputs (no measurable cost); the
report shows decline rates, FPR, FNR and calibration by group with intervals. Gaps remain, the
largest by education and age band, because proxies remain. The report is descriptive and says so;
it is not a compliance statement.

**How did you make serving faster without changing predictions?** Profiled: pandas overhead in
feature engineering dominated. A NumPy twin of the feature code plus the same model and fitted
calibrator gives bit-identical probabilities (checked at startup and on the full test split) and
raises one worker's throughput from ~100 to ~330 req/s.

**What would change in production?** A real feature store and event timestamps (temporal
validation), delayed-label joins for performance monitoring, a persistent registry, the rate limit
at the gateway, shadow deployment of challengers before promotion, and costs estimated from real
loss and margin data instead of an assumed ratio.
