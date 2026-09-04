# Architecture

This document is the reasoning, not the API. It is organised around the decisions
that were genuinely contested — what I chose, what I chose against, what it cost,
and where I would change my mind.

---

## 1. Why the agent has no language model in it

**Decision:** deterministic tool orchestration in the authorisation path; the LLM
only rewrites case memos, off the hot path, with a template fallback.

This is the decision most likely to be challenged, so it goes first.

An LLM-driven agent — one that reasons in text about which check to run next — is
the more fashionable answer, and for this problem it is the wrong one:

- **Latency.** Payment authorisation has a hard budget. This path decides in 55 µs
  at p50 and 111 µs at p99. An LLM call is 300–2,000 ms — three to four orders of
  magnitude more, on the critical path of every payment.
- **Reproducibility.** The audit log is the point of the system. A log that records
  a decision the system would not make again is not evidence of anything. A
  deterministic policy gives the same decision for the same transaction and the same
  state, every time, which is also what makes regression tests possible.
- **Cost at volume.** 917,792 transactions is a fortnight of traffic for a mid-sized
  merchant. Running any LLM on all of it, per transaction, is not a rounding error.
- **Attack surface.** Merchant names and category strings are attacker-controlled
  text. Putting attacker-controlled text into a model that decides whether to
  approve the attacker's payment is a prompt-injection vector with a direct payout.

So where *does* an LLM earn its keep? Where a human would otherwise write prose. A
held case needs a note an analyst can read; a declined merchant needs an explanation
support can send. That is ~0.1% of volume, latency-insensitive, and the failure mode
of a bad memo is a worse-written note rather than a wrong decision. `memo.py` does
exactly that, and falls back to deterministic templates whenever the API is absent
or erroring — a memo service outage must never affect a payment decision.

**Where I would change my mind:** an LLM belongs in the *analyst's* loop, not the
transaction's — summarising a day's held cases, clustering emerging fraud patterns
across memos, drafting the rule that would have caught a new pattern. All of those
are batch, none is in the auth path. That is the natural next version.

---

## 2. Expected cost, not a threshold

**Decision:** choose the action minimising `p·cost(action | fraud) + (1−p)·cost(action | legit)`
over four actions, rather than thresholding a score.

A threshold answers "is this fraud?" The business question is "what should we do,
given it might be?" Those differ whenever the amount varies, because the cost of
being wrong scales with the amount and the cost of an analyst does not.

Concretely, at a 2% probability of fraud: on ₹100 the cheapest action is to approve
(intervening costs more than the expected loss); on ₹100,000 it is not. One
threshold cannot express both. `tests/test_audit_and_policy.py::test_decision_threshold_moves_with_amount`
pins this.

The four actions matter as much as the arithmetic. Allow / step-up / review / block
is a *friction ladder*, and having the middle rungs is why Tribunal inflicts less
total friction than a blocking threshold while intervening 5.8× more often. A
two-action system has to answer a hard case by declining it.

**The catch, stated plainly:** this only works if `p` is a real probability. A model
trained with class reweighting to "fix" imbalance produces scores that rank well and
mean nothing, and multiplying those by rupees produces confident nonsense. Hence: no
reweighting, true priors preserved, isotonic calibration fitted on a held-out
window. The calibration curve in the README is not decoration — it is the
precondition for the policy being allowed to exist.

**Cost of the choice:** every number in `economics.py` is an assumption I made up
from plausible values, not measured. That is a real weakness. The mitigation is that
they are all in one file, named, and documented; the honest framing is that the
*ranking* of policies is robust to them while the absolute totals are not.

---

## 3. Point-in-time features, and delayed labels

**Decision:** stream state in timestamp order; `compute()` before `observe()`;
label-derived features held back by a 7-day dispute window.

Target leakage in fraud modelling is not exotic — it is one `groupby` away, and it
looks like ordinary code:

```python
df["card_mean_amt"] = df.groupby("cc_num")["amt"].transform("mean")   # sees the future
```

Nothing about that line looks wrong, and the resulting metrics look *better*, so
nothing alerts. The only reliable defence is to make it structurally impossible,
which is why features are built by streaming rather than by aggregation.

The delayed labels are the part I would defend hardest, because it is the part most
projects skip. At authorisation time you do not know a transaction is fraud. You
find out days or weeks later when the cardholder disputes it. A merchant-risk
feature computed from labels that had not arrived yet is a subtler leak than the
groupby, and a model built on it will underperform in production in a way that is
very hard to diagnose. `FeatureEngine` queues every label on a heap and releases it
only when the dispute window has elapsed.

**Cost:** ~4% relative PR-AUC, measured (`eval/leakage_ablation.py`). Also
complexity — the engine is the most intricate file in the project, and it is why
feature building is a Python loop rather than three lines of pandas. At 917,792 rows
that loop takes 18 seconds, which is a price worth paying.

**Trade-off I accepted:** the engine is stateful and in-memory. That is fine for a
batch build and for a single-process service, and wrong for a real deployment, where
trailing card state belongs in a shared low-latency store. `CardState` is
deliberately small and serialisable so that swap is mechanical.

---

## 4. Tools that produce evidence, not scores

**Decision:** each check returns an `Evidence` object — what it looked at, whether
it fired, how strongly, and one sentence a human can read — rather than a number
folded into a total.

The alternative is a weighted risk score. It is simpler and it is what most rule
engines do, and it makes the explanation problem unsolvable: once six signals have
been summed into 0.83, the only way back to "why" is a post-hoc attribution like
SHAP, which produces a *plausible* story rather than the actual reason.

Keeping evidence separate from the decision means the memo is not reverse-engineered
— it is a transcript. It also means the audit record can say which checks stayed
silent, which matters, because "we looked and found nothing" and "we never looked"
are different answers.

**Cost:** more code per check, and the temptation to let evidence vote on decisions,
which is exactly how the escalation rules ended up being worth ₹0 on this benchmark.
Escalation rules can only make a decision *stricter*, never looser — a rule that can
silently approve something the model flagged is an attack surface, not a safety net
— and their contribution is measured and reported separately rather than assumed.

---

## 5. Review as a budgeted resource

**Decision:** manual review admission runs through a daily capacity with an adaptive
threshold, driven by a proportional controller on *demand*.

Expected-cost minimisation, left alone, will route however many cases it likes to
review. Real queues hold a fixed number. A fixed score cutoff cannot hold a queue at
a fixed size either — volume and fraud rates move, and the queue starves or
overflows.

So a case is admitted when the *value of reviewing it* — the rupees review saves
over the best action available without an analyst — clears a bar, and the bar moves
daily to land admissions near capacity.

The controller steers on demand rather than admissions, and that distinction was a
bug I only found because a test failed. Admissions are clamped at capacity, so a
queue oversubscribed 10× looks identical to one exactly full: the controller sees
zero error and never raises the bar. Steering on the number of cases that *cleared
the threshold* gives it the signal it needs. (`test_controller_raises_the_bar_when_over_capacity`.)

When capacity runs out, a risky transaction falls back to the best action available
without an analyst — never to `allow`. Silently approving what you meant to
scrutinise, because a queue was full, is the worst possible failure mode, and it is
pinned by a test.

**Honest status:** on this benchmark, review is used **zero** times, because the
model is accurate enough that paying ₹120 for a human opinion is never optimal.

That is a statement about the price, not the machinery, and `eval/sensitivity.py`
pins where it changes: the policy asks for 5 reviews at ₹60 a case, 44 at ₹30, and
155 at ₹10. It also starts asking for reviews when the *cheap* intervention becomes
unreliable — 30 of them if step-up authentication only stops half of fraudsters.
So the queue is dormant here for a defensible reason, and the sweep says exactly
what would wake it. I am flagging it as currently-unexercised rather than
presenting it as a contributor to the headline number.

---

## 6. Hash-chained audit log

**Decision:** append-only JSONL, each record carrying the SHA-256 of its
predecessor.

The requirement is that a decision cannot be *quietly* altered after someone asks
about it. A plain log satisfies nobody; a database with an audit table has the same
problem one privilege level up. A hash chain means any edit or deletion invalidates
every subsequent record, and verification is a linear scan with no infrastructure.

**What it does not do**, stated so nobody over-reads it: it is tamper-*evident*, not
tamper-proof. Someone with write access can rewrite the entire file and recompute
the chain. Fixing that needs an external anchor — publishing the head hash somewhere
you do not control is the cheap version, and `AuditLog.head` exists for exactly that.

Approvals are sampled at 1% rather than logged in full: logging 18 months of
approvals proves nothing and costs storage, but logging *none* of them makes it
impossible to audit what was let through for bias. Card numbers are masked to the
last four before writing.

---

## 7. Model choice

Gradient-boosted trees (`HistGradientBoostingClassifier`), not a neural network and
not logistic regression.

- Tabular data with mixed types, ~30 features, 900k rows: this is the regime where
  boosted trees still win, and they train in 23 seconds on two cores.
- Native handling of NaN matters here, because missing history is *informative* —
  a card's first-ever transaction genuinely has no velocity, and encoding that as
  zero would be a lie. The model learns from the absence directly.
- Native categorical support avoids one-hot expansion of merchant category.
- `sklearn` only — no XGBoost or LightGBM dependency. It is a smaller install for a
  reviewer, and the accuracy difference at this scale is not the bottleneck.

Deliberately no deep learning. There is no sequence model over a card's transaction
history, which is the one place a neural approach would plausibly beat this — and
which is the next thing I would build, with the trailing-state store from §3 as its
input.

---

## 8. What I would do differently with more time

- **Measure the cost model instead of assuming it.** Every parameter in
  `economics.py` is a guess at a real quantity: chargeback fees, step-up
  abandonment, fully-loaded analyst cost. `eval/sensitivity.py` establishes that the
  *ranking* survives all 41 settings tested, so the architectural conclusion is
  safe — but the rupee totals move by a factor of five across that range, and the
  size of the advantage depends most on step-up effectiveness. Those are the two
  numbers I would measure first with access to real data.
- **Drift detection.** The model is trained once and never retrained. Fraud is
  adversarial; a production system needs population-stability monitoring on features
  and scores, and a retraining trigger.
- **A real feedback loop.** Analyst verdicts on reviewed cases are the highest-value
  labels available and currently go nowhere.
- **Better data.** Every limitation in the README traces back to the simulator.
  Validating on a real labelled stream would change more than any modelling change
  in this list.
- **Sequence modelling** over per-card history, per §7.
