# Hackathon template — Red vs Blue

You implement either the **red** agent (extracts KPIs from a document and
smuggles hallucinations) or the **blue** agent (judges each KPI as correct
or hallucinated). The orchestrator runs everyone against everyone.

## Setup

```bash
pip install -e ".[dev,llm]"
cp .env.example .env  # add your Groq key (provided on the day)
pytest                # smoke test the baseline pipeline
```

## What to change

- **Red side:** edit `red/submission.py`. Subclass `RedAgent` and bind
  `agent` to your instance. Look at `red/baseline.py` for the contract.
- **Blue side:** edit `blue/submission.py`. Subclass `BlueAgent`.

You can split your code across as many files as you want inside `red/` or
`blue/`. The orchestrator only imports `agent` from `submission.py`.

Shared types and the scorer live in `shared/`. Don't modify them.

## Run a local match

```bash
python scripts/run_match.py examples/finance_short/example_01
```

This runs your red against your blue on a synthetic example and prints the
score breakdown. Use it to debug before pushing.

## The contract

- **Red** outputs a `RedExtraction`: just a list of `KPI`s. You do **not**
  flag which ones you tampered with — the scorer derives each KPI's truth
  from the ground truth (see `shared/scoring.py`). Quota (GT-relative):
  reproduce ≥50% of the distinct GT KPIs, add ≤25% of the GT count as
  distinct hallucinations.
- **Blue** outputs a `BlueJudgment`: a binary verdict (`correct` or
  `hallucinated`) for every KPI id the red emitted.
- **Scoring:** blue is scored as **macro-F1** over the two classes
  `{correct, hallucinated}`, so `blue_score = (F1_correct + F1_hallucinated)/2`
  and `red_score = 1 − blue_score`. Red then takes **additive, progressive
  quota penalties** (proportional to how far out of quota it is), clamped to
  `[0, 1]`.

## Hallucination taxonomy

The scorer classifies every red KPI against the ground truth, by priority:

| Type         | Rule (vs the ground truth)                                    |
| ------------ | ------------------------------------------------------------- |
| `correct`    | reproduces a GT row on **every** field                        |
| `value`      | same identity, wrong number — some GT row matches all metadata, only the value differs |
| `metadata`   | right value, wrong context — the value exists in GT but unit/scope/period/name differ |
| `fabricated` | neither the value nor the full metadata appears in GT         |
