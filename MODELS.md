# Models

Eight in total: four predictive models, two reference baselines they are scored
against, and two unsupervised models for the neighbourhood layer.

Each predictive model is fitted twice, once for each task, so the performance
table carries twelve rows: six approaches on each of two tasks.

## The two supervised tasks

| Task | Question | Label |
|---|---|---|
| **Level** | Will this authority be in the worst national fifth next year? | `next_year >= cutoff`, where the cutoff is the 80th percentile **of the training years only** |
| **Escalation** | Will family homelessness rise by 25% or more next year? | `next_year / this_year >= 1.25`, null where this year is zero |

Both are binary. Features come from year *t*, labels from year *t+1* for the
same authority. 1,686 rows, 83 features.

## 1. Logistic regression

```python
Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale",  StandardScaler()),
    ("clf",    LogisticRegression(max_iter=5000, C=0.3, random_state=0)),
])
```

The interpretable reference. L2 penalty at `C=0.3`, which is fairly strong
regularisation, because 83 features against roughly 1,100 training rows will
otherwise overfit. Imputation and scaling sit inside the pipeline so they are
fitted on training folds only and never see the test year.

This is also the model behind the "what moves the escalation score" chart. Its
standardised coefficients give a direction and a magnitude per feature, which
is what an officer needs in order to disagree with a score.

On the harder task it is the strongest of the fitted models. With this sample
size the intervals overlap heavily, so that ordering should not be over-read.

## 2. Histogram gradient boosting

```python
HistGradientBoostingClassifier(
    max_depth=3, max_iter=300, learning_rate=0.06,
    l2_regularization=1.0, min_samples_leaf=15, random_state=0)
```

Depth 3 and a minimum of 15 samples per leaf keep it deliberately shallow. The
dataset is small and the positive class is thin, so a deeper model would fit
noise.

Chosen partly because it **handles missing values natively**. Fourteen per cent
of the design matrix is missing, and the pattern is informative rather than
random: an authority that filed no return is missing everything for that year.
Imputing those would invent data. The tree splits route missing values down
whichever branch reduces loss, which is closer to honest.

## 3. Gradient boosting, isotonically calibrated

```python
CalibratedClassifierCV(HistGradientBoostingClassifier(...),
                       method="isotonic", cv=3)
```

The model the dashboard and the reported numbers use.

Boosted trees rank well but their raw scores are not probabilities. Whoever
acts on a triage score reads it as one, so it has to be one. Isotonic rather
than Platt scaling because there is no reason to assume the miscalibration
takes a sigmoid shape.

It moves the Brier score on the level task from **0.19 to 0.089**, which is the
single clearest improvement any model makes anywhere in this project. The
ranking barely changes; what changes is whether the number can be believed.

## 4. Neural network with Monte-Carlo dropout

```python
Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale",  StandardScaler()),
    ("clf",    MCDropoutMLP(hidden=(64, 32), dropout=0.3)),
])
```

A small feed-forward network in PyTorch: 83 inputs, two hidden layers of 64 and
32 units, ReLU, and dropout after every layer. AdamW with weight decay, a
reweighted loss for the small positive class rather than resampling (which
would distort the calibration), and early stopping on a random slice of the
training years.

It holds its own without winning: **0.93** on the level task and **0.59** on
escalation, in both cases inside the confidence interval of the best model for
that task. On this sample size that is the honest reading, and it is worth
saying rather than presenting a fractional difference as a result.

What it adds that the other three cannot is **uncertainty**.

Dropout is normally switched off at prediction time. Leaving it on and
averaging 100 stochastic forward passes makes the network an approximate
Bayesian model, in the sense of Gal and Ghahramani (2016). Each authority then
comes with a spread as well as a score, and `predict_uncertainty` returns it.

That distinction matters operationally. A ranked list cannot express the
difference between "the model is confident this authority is fine" and "the
model has no idea about this authority", and those two cases deserve different
responses from a service. The same reasoning drives selective prediction, where
a model that abstains where it is uncertain is more useful than one that always
answers.

The architecture is deliberately small. With 83 features and roughly 1,100
training rows, anything wider memorises the training years, which is exactly
the failure the prospective backtest is designed to expose.

## 5. Persistence baseline

Not a fitted model. For the level task it is simply this year's rate; for
escalation, this year's one-year change. It represents what a service already
has without any analytics: sort the column and look at the top.

It is in the results table because two of the three findings come from it.

- On **level** it scores 0.90, which is close enough to the model's 0.95 that
  building machine learning for that question is hard to justify.
- On **escalation** it scores **0.35 [0.26–0.44]**. The interval excludes 0.5,
  so it is significantly *worse* than chance. Rises mean-revert, and a service
  triaging on "it went up last year" is systematically working the wrong list.

## 6. Prevalence baseline

```python
DummyClassifier(strategy="prior")
```

Predicts the base rate for everyone. It fixes the floor for PR-AUC and for the
hit rate in a fixed-length list, which is the comparison that matters
operationally. Without it, "30% of the flagged authorities escalated" has no
meaning; against a 15% base rate it means twice the yield.

## 7. k-means, for neighbourhood segments

```python
KMeans(n_clusters=5, n_init=50, random_state=0)
```

Runs on nine standardised deprivation domain and sub-domain scores across
Colchester's 105 LSOAs. IMD itself is excluded from the features, because it is
a weighted average of the others and would dominate the geometry.

`k` is chosen by silhouette score, with one documented constraint. `k=2` wins
outright at 0.365, but it only recovers "deprived" and "not deprived", which
the service already knows. Selection is therefore restricted to `k >= 4`, and
`k=5` wins within that range at 0.250. The pipeline prints the full silhouette
curve so the trade-off is visible rather than buried.

Segments are named for the domain on which they deviate most from the
Colchester average, walked in order of child income deprivation, skipping any
domain already used so no two segments share a name.

## 8. Isolation Forest, for anomaly detection

```python
IsolationForest(n_estimators=500, contamination=0.10, random_state=0)
```

Same nine features, same 105 neighbourhoods. It flags neighbourhoods that are
unusual **for Colchester**, which is not the same as deprived in absolute
terms. A high IMD score is not news; a neighbourhood with moderate overall
deprivation but a barriers-to-housing score three standard deviations above the
local mean is.

Eleven are flagged. Five sit in the less deprived half of England and between
them contain 1,337 children aged 0–15, which is the case against triaging on a
headline deprivation ranking.

Contamination is fixed at 10% rather than tuned. With 105 points and no ground
truth there is nothing honest to tune it against, so it is stated as a choice
and the ranked scores are published alongside the binary flag.

## What is deliberately absent

No heavy hyperparameter search. Every extra degree of freedom spent against a
held-out year of 262 authorities buys noise rather than signal.

No resampling to correct the class imbalance. It distorts calibration, and
calibration is the thing this project treats as non-negotiable. The loss is
reweighted instead.

No stacking or ensembling. The differences between these models sit inside
their confidence intervals, so combining them would be fitting to noise and
presenting it as an improvement.

The network is kept small for the same reason. Depth here would buy training
accuracy and nothing else, and the prospective backtest is designed to catch
exactly that.

## Validation, applied identically to all of them

Fitted on outcome years up to 2023–24, scored once on 2024–25. Nothing is tuned
on the test year. ROC-AUC carries a 2,000-draw percentile bootstrap interval,
alongside PR-AUC, hit rate in a list of 30, and Brier score.

## Where this comes from

The constructors above are the ones in `fhew.models`, and the descriptions
beside them are the same strings the dashboard prints, read from that module
rather than retyped. The specifications published in
`outputs/models/specifications.json` are introspected from the fitted
pipelines, so a hyperparameter cannot change without the published description
changing with it.

See [DATASET.md](DATASET.md) for the data these run on, and the
[Layout](README.md#layout) section of the README for how the code is arranged.
