# 5-minute pitch — script

Timings are for a 5:00 recording. Screen-share the repo and `reports/` figures;
talk over them. Do not read this verbatim — it is the argument, in order.

---

### 0:00 – 0:35 · The problem, sharply

> Most fraud submissions you'll watch today are a classifier: a score, a threshold,
> an AUC. I want to start with the number that made me build something else.
>
> *[show the policy comparison chart]*
>
> That red bar is a set of sensible hand-written fraud rules — the kind a team has
> before it has a model. On my held-out data it doesn't underperform. It **loses
> ₹156,000**. It's worse than approving every single transaction, because it catches
> 81% of fraud by declining 3,607 legitimate customers to do it.
>
> A model that only optimises for catching fraud can't see that. So Tribunal isn't
> built around a score. It's built around a decision.

---

### 0:35 – 1:35 · What it does differently

> Four things a classifier doesn't handle.
>
> **One: a score is not a decision.** At a 2% chance of fraud, declining a ₹200
> transaction is a bad trade and declining a ₹90,000 transaction is a good one. One
> threshold cannot express both. So instead of thresholding, Tribunal prices four
> actions in rupees — approve, challenge with an OTP, hold for review, decline — and
> picks the cheapest.
>
> **Two: the middle rungs matter.** Here's my favourite number in the project.
> *[point at the table]* Against a plain blocking threshold, Tribunal intervenes on
> **5.8× more good customers** — and inflicts **19% less** friction on them in total.
> Because most of those extra interventions are an OTP, not a decline. It's not
> choosing whether to intervene. It's choosing how.
>
> **Three: review is a fixed number of people**, so admission runs through a daily
> budget with a controller, not an unbounded queue.
>
> **Four: somebody will ask why**, later. Every decision writes a memo from the
> evidence that actually fired, into a hash-chained log.

---

### 1:35 – 2:15 · Show it working

> *[terminal: POST /decide, then the memo]*
>
> That's the live path. Six deterministic checks, a calibrated model, an
> expected-cost decision, and a memo built from the evidence — not reverse-engineered
> from a SHAP plot afterwards.
>
> Note the last line: **"checks that did not fire."** "We looked and found nothing"
> and "we never looked" are different answers to a regulator, so both get recorded.
>
> *[run the audit verify]* — 1,699 records, chain intact. Change one past decision
> and every hash after it breaks.
>
> And it decides in **55 microseconds** at p50. There's no LLM in this loop, and
> that's deliberate — I'll come back to it.

---

### 2:15 – 3:30 · The part I'd actually defend: what I found that I didn't want to

> Three things, because a result with no failures in it hasn't been looked at hard
> enough.
>
> **My second most important feature was learning the opposite of reality.**
> *[importance chart]* The card's own prior fraud rate scored second — worth 0.30
> PR-AUC. Then I checked its direction. Cards in the top three quintiles of prior
> confirmed fraud have **exactly zero** subsequent fraud. Single-feature AUC: 0.10 —
> strongly predictive, sign reversed. In the simulator each card is defrauded once,
> so "was defrauded before" means "is safe now." In production it means the
> opposite: that's a reissued card belonging to a targeted customer.
>
> I deleted it. Cost me 0.02 PR-AUC and ₹21,000. That's the right price for not
> shipping a model that under-protects the customers who've already been hurt once.
>
> **Geolocation is noise here, so it doesn't get a vote.** I built impossible-travel
> detection, then measured it: AUC 0.494 — a coin flip. The simulator scatters
> merchant coordinates independently of fraud. The check still runs and reports, but
> it's barred from escalating anything, with the measurement in the code next to the
> flag.
>
> **And a bug that made my model look better.** My time features were built with a
> conversion that compressed 18 months into 47 seconds — every velocity window
> covered the whole dataset. AUC went *up*. Nothing alerted, because leaks always
> flatter you. I found it plotting distributions and noticing two columns were
> identical. There's a regression test for it now, and 28 tests total.

---

### 3:30 – 4:20 · Is the complexity justified? Partly — here's where.

> Honest answer on this benchmark: the expected-cost policy beats a plain threshold
> by only **2.5%**, and the review queue is used **zero** times. The model is so
> accurate here that paying an analyst ₹120 is never worth it.
>
> That's a fact about the simulator, not about fraud. So I measured where the
> decision layer starts to matter — by building genuinely weaker models and re-running
> every policy against each.
>
> *[sweep chart]*
>
> At PR-AUC 0.29 the decision layer is worth **+163%**. At 0.96 it's worth 3%. The
> weaker the model, the more it matters — because uncertainty is exactly when what
> you *do* about a score matters more than the score.
>
> Real card-fraud models don't run at 0.97. This benchmark is the regime where my
> architecture matters **least**, and it still wins.

---

### 4:20 – 5:00 · Why no LLM, and close

> Last thing, because it's the decision you'll want to challenge. There's no language
> model in the decision path, and that's a choice, not a gap.
>
> This code sits in the authorisation path of a payment. It decides in 55
> microseconds; an LLM call is three orders of magnitude slower. It's reproducible —
> which is the only thing that makes the audit trail worth having. And merchant names
> are attacker-controlled text, so putting them into a model that decides whether to
> approve the attacker's payment is a prompt-injection vector with a direct payout.
>
> The LLM earns its place where a human would otherwise write prose: the case note
> for the 0.1% of transactions that reach an analyst. Off the hot path, with a
> template fallback, because a memo outage must never affect a payment decision.
>
> Everything reproduces from a seed. There's no notebook. The failures are in the
> README, not just the wins.
>
> **Every decline gets a reason. Every reason gets a receipt.**

---

## Numbers to have memorised

| | |
|---|---|
| Test window | 160,139 txns, 92 days, 1,055 fraud, ₹560,519 at risk |
| Tribunal | ₹1,040,422 saved · 99.3% fraud value stopped · recall 0.964 |
| vs threshold | ₹1,014,614 · 5.8× more customers touched · 19% *less* friction |
| Hand rules | −₹156,541 — worse than doing nothing |
| Model | ROC-AUC 0.9989 · PR-AUC 0.9679 · Brier 0.00065 |
| Latency | 55 µs p50 · 106 µs p99 · 16,904 txns/s |
| Deleted feature | worth 0.30 PR-AUC, AUC 0.102 — inverted |
| Sweep | +163% uplift at PR-AUC 0.29 → +3% at 0.96 |
| Tests | 35 passing |
| Sensitivity | ranking holds at all 41 cost settings; review pays below ₹60/case |

## Likely panel questions

**"Why is your precision only 0.74?"** — Because precision counts interventions and
the business pays in rupees. Most of my extra interventions are OTP challenges, not
declines; total friction inflicted is *lower* than the higher-precision threshold
policy. I'd rather challenge four good customers than decline one.

**"Your cost model is made up — why should I believe the rupee figures?"** — You
shouldn't believe the totals, and I say so in the README. I swept all seven
parameters across plausible ranges, 41 settings: the *ranking* never flips, against
a threshold baseline I let tune on the test set. The margin ranges from +4% down to
+0.2%, and the floor is the case where step-up only stops half of fraudsters — so if
you gave me one number from your business to measure first, it'd be OTP
effectiveness.

**"Your AUC is 0.99 — isn't that suspicious?"** — Yes, and I chased it. It's the
simulator: fraud is high-amount and concentrated at night. I also found and fixed a
generator defect that made it worse, deleted a feature that was exploiting an
artifact, and ran the whole evaluation again across a ladder of deliberately weaker
models. Treat every absolute number as an upper bound; the policy comparisons are
what the project is about.

**"Why is review never used?"** — Because an analyst costs ₹120 and this model is
accurate enough that the uncertainty isn't worth that. I swept the price: the policy
wants 5 reviews at ₹60 a case, 44 at ₹30, 155 at ₹10. So it's a capacity-planning
answer, not a broken component — manual review pays on this traffic below about ₹60.
It also wakes up when step-up is unreliable: if OTP only stops half of fraudsters,
it starts asking for 30 human reviews.

**"Where does it fail?"** — 38 frauds worth ₹3,770 get through: small, at familiar
merchants, in normal hours, inside the cardholder's usual range. Catching them would
cost more in false declines than they're worth. And my calibration is
under-confident in the 0.5 band, which biases toward over-intervening in exactly the
ambiguous range.

**"What would you build next?"** — Measure the cost model instead of assuming it;
those parameters are the biggest source of uncertainty in my rupee figures. Then a
sequence model over per-card history, and an LLM in the *analyst's* loop — clustering
held cases to surface emerging patterns — not in the transaction's.
