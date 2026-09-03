# Geolocation carries no fraud signal in this benchmark

Referenced from `tribunal/tools/heuristics.py:GEO_IS_DIAGNOSTIC`.

Impossible-travel detection is among the strongest signals in real card-present
fraud, so the check is built and it runs. It is barred from escalating a decision
here because on this data it measures nothing. The measurements, on a 200,000-row
sample of the transaction table:

| Signal | ROC-AUC vs `is_fraud` |
|---|---:|
| Distance from cardholder's home region | **0.4942** |
| Transaction amount (for comparison) | 0.8388 |
| Night-time flag (for comparison) | 0.8036 |

Distance from home, by label:

| | count | mean km | median km | p90 km |
|---|---:|---:|---:|---:|
| legitimate | 198,556 | 76.4 | 78.6 | 113.1 |
| fraud | 1,444 | 75.8 | 78.1 | 114.7 |

Implied travel speed between consecutive transactions is equally flat: median
4,011 km/h for legitimate transactions against 3,462 km/h for fraud. Both are
absurd values in absolute terms, which is the tell — the simulator draws each
merchant's coordinates from a fixed radius around the customer's home,
independently of the label, and consecutive merchants are unrelated points rather
than a path a person walked.

A geo escalation rule on this data could therefore only manufacture false
positives. On real data, flip `GEO_IS_DIAGNOSTIC` and re-run this measurement
before trusting it.

Reproduce: `eval/` has no dedicated script for this; the numbers come from
`roc_auc_score(sample.is_fraud, distance)` using `tribunal.features.haversine_km`
over `data/transactions.parquet`.
