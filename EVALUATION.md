# 📊 Predictive Evaluation Results

This document summarises the first full predictive evaluation run of the Temporal Graph Network (TGN) model and baseline systems.  
Evaluation uses the **predictive emergence objective**: can the model’s top-K predictions at time *t* correctly anticipate which items become emergent trends in the following Δ horizon window?

---

## ⚙️ Setup

- **Events file**: `data/events.jsonl`
- **Windows**: 60 minutes
- **Future horizon (Δ)**: 120 minutes
- **Emergence threshold**: ≥8 mentions
- **Minimum growth**: ≥4 mentions vs. past window
- **Support filter**: ≥1 historical mention
- **K values**: 5, 10, 20
- **Evaluation windows**: 127

Command used:

```bash
python -m model.evaluation.baseline_eval \
  --events data/events.jsonl \
  --outdir data/eval_predictive \
  --window-minutes 60 \
  --horizon-minutes 120 \
  --emergence-threshold 8 \
  --min-growth 4 \
  --min-support 1 \
  --k 5 10 20
```

Windows (single-line command):

```bash
python -m model.evaluation.baseline_eval --events data/events.jsonl --outdir data/eval_predictive --window-minutes 60 --horizon-minutes 120 --emergence-threshold 8 --min-growth 4 --min-support 1 --k 5 10 20
```
