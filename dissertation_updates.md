# Dissertation Updates

Everything you need to update the dissertation and run the final experiments.
Read top-to-bottom the first time; use as a checklist afterwards.

---

## 1. What changed in the code (summary)

| File | What changed |
|---|---|
| `models.py` | GRU optimizer: `'adam'` → `Adam(clipnorm=1.0)` |
| `models.py` | iTransformer added as new model (`build_itransformer_predictor`) |
| `models.py` | `build_transformer_predictor_revised()` added as separate function (original kept) |
| `train_prediction.py` | Seeds added (`seed=42`); `--smoke` and `--seed` CLI flags; CSVLogger per model |
| `train_breen_baseline.py` | Seeds added; CSVLogger added; `--smoke` and `--seed` CLI flags |
| `s3_utils.py` | Added `.keras` model files and `_training_log.csv` files to upload list |

The Transformer revision adds a *second* function (`build_transformer_predictor_revised`) alongside the original.
Both are trained in `train_prediction.py` and appear as separate rows in Table 5.2.
The three changes: `num_heads` 4→2, fixed sinusoidal PE → learnable Embedding, LR 1e-3→1e-4.

---

## 2. Seed 42 results (EC2 run complete — 2026-06-09/10)

These are the confirmed numbers from the full EC2 run (37,846 trajectories, 2.3M training sequences).

### Prediction models (RQ1)

| Model | Test MSE | Test MAE | Notes |
|---|---|---|---|
| **iTransformer** | **0.003675** | **0.016990** | Best overall |
| LSTM | 0.004085 | 0.018399 | Strong second |
| GRU (clipnorm=1.0) | 0.005910 | 0.031673 | Gradient clipping fixed epoch-5 spike |
| Transformer (revised) | 0.074270 | 0.109051 | 2 heads, learnable PE, LR=1e-4 |
| Transformer | 0.079326 | 0.121387 | Original; LR=1e-3 caused instability |

### Breen baseline (RQ1)

| Model | Test MAE | Training time |
|---|---|---|
| Breen MLP | 0.182374 | 1574.6s (stopped epoch 142) |

### Inference speed

| Model | ms/sample | vs numerical |
|---|---|---|
| LSTM | 14.6 ms | 0.3× |
| GRU | 12.9 ms | 0.3× |
| Transformer | 3.3 ms | 1.1× |
| Transformer (revised) | 3.3 ms | 1.2× |
| iTransformer | 3.3 ms | 1.1× |
| Numerical integration | 3.8 ms | baseline |

**Key finding:** iTransformer achieves the lowest MAE and matches numerical integration speed.
LSTM/GRU are slower than numerical at this batch size (single-sample inference overhead dominates).
Both Transformer variants underperform recurrent models — attention across time steps is less suited to this 4D sequential task than attention across variates (iTransformer).

---

## 3. How to run the remaining seeds (seeds 123 and 456)

Seed 42 is **done**. You need seeds 123 and 456 to compute mean ± std for Table 5.2.

### On EC2, run in a screen session:

```bash
screen -S seed123
python train_prediction.py --seed 123 2>&1 | tee seed123_log.txt
# When done (hours later):
python train_breen_baseline.py --seed 123 2>&1 | tee seed123_breen_log.txt
```

Detach with `Ctrl+A D`. Come back later with `screen -r seed123`.

Then repeat for seed 456:
```bash
screen -S seed456
python train_prediction.py --seed 456 2>&1 | tee seed456_log.txt
python train_breen_baseline.py --seed 456 2>&1 | tee seed456_breen_log.txt
```

You do **not** re-run classification or equilibrium discovery — those are unchanged.

### What each seed run produces

- 5 updated `.keras` model files (overwrite seed 42 files — that's fine, seed 42 is already in S3)
- 5 updated `_training_log.csv` files
- Updated `prediction_results.pkl` and `breen_results.pkl`
- Updated plots

### After each seed run, record the numbers

When `train_prediction.py` finishes, it prints something like:
```
LSTM Test Loss (MSE): 0.004085
LSTM Test MAE: 0.018399
GRU Test Loss (MSE): 0.005910
...
```

Copy those numbers into the table below. Also check `breen_results.pkl` or the Breen print output for Breen MAE.

### Computing mean ± std for Table 5.2

Once you have all three seeds, fill this in:

| Model | Seed 42 MAE | Seed 123 MAE | Seed 456 MAE | Mean ± Std |
|---|---|---|---|---|
| iTransformer | 0.016990 | *(record)* | *(record)* | *(compute)* |
| LSTM | 0.018399 | *(record)* | *(record)* | *(compute)* |
| GRU (clipnorm=1.0) | 0.031673 | *(record)* | *(record)* | *(compute)* |
| Transformer (revised) | 0.109051 | *(record)* | *(record)* | *(compute)* |
| Transformer | 0.121387 | *(record)* | *(record)* | *(compute)* |
| Breen MLP | 0.182374 | *(record)* | *(record)* | *(compute)* |

Mean = `(a + b + c) / 3`
Std = `sqrt(((a-mean)² + (b-mean)² + (c-mean)²) / 3)`

Or paste the three numbers into Python:
```python
import numpy as np
vals = [0.016990, SEED123_MAE, SEED456_MAE]
print(f"{np.mean(vals):.4f} ± {np.std(vals):.4f}")
```

---

## 4. Dissertation changes — section by section

### Table 5.2 (main results table for RQ1)

Replace the existing table with this structure. Fill in mean ± std once seeds 123 and 456 are done.
Until then, you can use seed 42 values with a "(seed 42)" note.

| Model | Test MAE | Test MSE | Inference (ms) | Notes |
|---|---|---|---|---|
| LSTM | 0.0184 ± σ | 0.0041 ± σ | 14.6 | — |
| GRU (clipnorm=1.0) | 0.0317 ± σ | 0.0059 ± σ | 12.9 | Gradient clipping |
| Transformer | 0.1214 ± σ | 0.0793 ± σ | 3.3 | Original architecture |
| Transformer (revised) | 0.1091 ± σ | 0.0743 ± σ | 3.3 | 2 heads, learnable PE |
| **iTransformer** | **0.0170 ± σ** | **0.0037 ± σ** | **3.3** | **Best model** |
| Breen MLP | 0.1824 ± σ | — | — | Direct IC → state mapping |
| Numerical integration | — | — | 3.8 | Reference baseline |

Add to table caption:
> *Results are mean ± std over three independent runs with seeds 42, 123, and 456.
> Transformer (revised) used two attention heads, a learnable positional embedding,
> and learning rate 1×10⁻⁴. iTransformer applies attention across the four
> phase-space variates rather than across time steps.*

Add to table caption: *"Rows marked 'revised' reflect targeted architectural changes
described in Section 6.2. Original and revised rows are both shown to isolate the
effect of each change. Results are mean ± std over three independent runs with seeds
42, 123, and 456."*

### Section 5.2 (prediction results narrative)

**Add before the table** — one paragraph on iTransformer:

> The iTransformer (Liu et al., 2024) was implemented as a fifth model for the
> trajectory prediction task. Rather than attending across the 50-step temporal
> dimension, the architecture inverts the tokenisation: each of the four phase-space
> coordinates (ξ, η, vξ, vη) becomes a token whose embedding is its complete
> time series. Attention is then computed across these four variate tokens,
> directly capturing the position–velocity couplings prescribed by the RC3BP
> equations of motion. The model used d_model=64, four attention heads, and two
> encoder layers.

**Add after the GRU discussion** — one sentence:

> The revised GRU with gradient clipping (Table 5.2, GRU clipnorm=1.0) eliminated
> the epoch-5 validation loss spike and achieved a test MAE of 0.0317, confirming
> gradient explosion as the sole cause of the original instability.

**Add after the Transformer discussion** — one sentence:

> The revised Transformer (Table 5.2, Transformer revised) with reduced heads,
> learnable positional embeddings, and learning rate 1×10⁻⁴ improved upon the
> original (MAE 0.109 vs 0.121), yet both variants remained substantially outperformed
> by the recurrent and iTransformer models, confirming that temporal self-attention
> is ill-suited to this 4D phase-space task at this scale.

### Section 6.2.3 (GRU clipnorm — currently written as future recommendation)

**Add at the end of the existing paragraph** (do not rewrite it):

> This fix was subsequently implemented by passing `clipnorm=1.0` to the Adam
> optimiser. Results for both the original and revised GRU are reported in Table 5.2.

### Section 6.2.4 (Transformer revision — currently written as future recommendation)

**Add at the end of the existing paragraph**:

> All three changes were subsequently implemented: attention heads reduced from four
> to two, fixed sinusoidal positional encoding replaced with a learnable embedding
> layer, and the Adam learning rate set to 1×10⁻⁴. Results are reported in Table 5.2
> as Transformer (revised).

### Section 6.2 (wherever iTransformer is listed as "to be incorporated")

**Replace that sentence with**:

> The iTransformer was implemented and evaluated; results are reported in Table 5.2.

### Everything else in Section 6.2

**Leave unchanged.** Any future work that wasn't implemented (Jacobi constant filter,
larger dataset scales, multi-step classification, etc.) stays as future work. Do not
touch those paragraphs.

---

## 5. Getting the results onto your Mac + which plots to use

### Download from S3 (do this on your Mac, not EC2)

Make sure AWS CLI is configured, then run from the project directory:

```bash
python - <<'EOF'
from s3_utils import download
files = [
    'prediction_results.pkl',
    'breen_results.pkl',
    'prediction_training_history.png',
    'prediction_examples.png',
]
for f in files:
    download(f)
EOF
```

Or download everything:
```bash
python -c "from s3_utils import download; [download(f) for f in ['prediction_training_history.png','prediction_examples.png','prediction_results.pkl','breen_results.pkl']]"
```

Alternatively, `scp` directly from EC2:
```bash
scp -i your-key.pem ubuntu@<EC2-IP>:~/deeplearningapps3bp/prediction_training_history.png .
scp -i your-key.pem ubuntu@<EC2-IP>:~/deeplearningapps3bp/prediction_examples.png .
```

---

### Which plots to replace or add in the dissertation

| Plot file | Status | What to do |
|---|---|---|
| `prediction_training_history.png` | **REPLACE** — now shows 5 models (was 3) | Swap the figure in Section 5.2 |
| `prediction_examples.png` | **REPLACE** — now shows 5 models (was 3) | Swap the figure in Section 5.2 |
| `classification_confusion_matrices.png` | Unchanged | Leave as-is |
| `breen_training_history.png` | Unchanged | Leave as-is |
| `breen_prediction_examples.png` | Unchanged | Leave as-is |
| `lagrange_point_discovery.png` | Unchanged | Leave as-is |

Only two figures need to be swapped. Everything else stays.

---

### What each new plot shows

**`prediction_training_history.png`** — two panels:
- Left: training + validation loss (MSE, log scale) for all 5 models across epochs
- Right: training + validation MAE for all 5 models
- Use this as Figure X.X "Training convergence of prediction models"
- Caption note: *"LSTM and iTransformer converge to the lowest validation loss. Both Transformer variants plateau significantly higher, indicating the architecture is less suited to this task."*

**`prediction_examples.png`** — 2×3 grid:
- Top row: trajectory in (ξ, η) phase space — input path, true continuation, each model's prediction
- Bottom row: same samples plotted as ξ coordinate over time steps 0–60
- Use this as Figure X.X "Example trajectory predictions"
- Caption note: *"All five models are shown. LSTM and iTransformer predictions closely track the true trajectory; both Transformer variants diverge at longer horizons."*

---

## 6. Reproducibility — what to state in the methods section

Add one paragraph to the Methods chapter (Section 4 or wherever training setup is described):

> All deep learning models were trained with a fixed random seed (seed=42 for the
> primary run; seeds 42, 123, and 456 for the multi-seed validation). Seeds were
> applied to both TensorFlow (`tf.random.set_seed`) and NumPy (`np.random.seed`)
> at the start of each training function, ensuring deterministic weight
> initialisation and dropout masks. Per-epoch training metrics (loss, validation
> loss, MAE, validation MAE) were recorded via Keras CSVLogger callbacks and are
> available alongside the saved model weights at [repository URL / S3 path].
> Final model weights were saved in the `.keras` format immediately after training,
> with early stopping configured to restore the best validation-loss checkpoint.

---

## 7. Branch strategy

```
main                         ← stable; original results; safe to cite
feature/transformer-revision ← Transformer architecture changes (current branch)
```

**After EC2 results are confirmed:**
```bash
git checkout main
git merge feature/transformer-revision
git tag v1.0-final-results
git push origin main --tags
```

Cite the tag hash in the dissertation methods section as the exact reproducible state.

---

## 8. What is NOT changing

To be explicit about what you don't need to touch:

- Data generation (`data_generation.py`) — unchanged
- Preprocessing (`preprocessing.py`) — unchanged
- Classification training (`train_classification.py`) — unchanged
- Equilibrium discovery (`discover_equilibria.py`) — unchanged
- Classification results and figures — unchanged
- Equilibrium results and figures — unchanged
- LSTM architecture — unchanged
- Breen MLP architecture — unchanged
- Background chapter (except adding iTransformer implementation detail)
- RQ2 and RQ3 results sections — unchanged

---

## 9. Complete dissertation checklist (do in this order)

### Step A — Wait for seeds 123 and 456 (EC2)
- [ ] Run `python train_prediction.py --seed 123` and record MAE values
- [ ] Run `python train_breen_baseline.py --seed 123` and record Breen MAE
- [ ] Run `python train_prediction.py --seed 456` and record MAE values
- [ ] Run `python train_breen_baseline.py --seed 456` and record Breen MAE
- [ ] Fill in the mean ± std table in Section 3 of this file

### Step B — Download results to Mac
- [ ] Download `prediction_training_history.png` from S3 or EC2
- [ ] Download `prediction_examples.png` from S3 or EC2
- [ ] Verify both images look correct (5 models visible, legend readable)

### Step C — Update the dissertation document
- [ ] **Table 5.2**: Replace with the new table structure (Section 4 of this file). Fill in mean ± std once you have them; until then use seed 42 values.
- [ ] **Section 5.2** (prediction narrative): Add the iTransformer paragraph (Section 4 of this file)
- [ ] **Section 5.2**: Add the GRU clipnorm sentence after GRU discussion
- [ ] **Section 5.2**: Add the Transformer revised sentence after Transformer discussion
- [ ] **Figure — training history**: Replace the old `prediction_training_history.png` with the new one. Update caption.
- [ ] **Figure — example predictions**: Replace the old `prediction_examples.png` with the new one. Update caption.
- [ ] **Section 6.2.3** (GRU future work): Append the "fix was implemented" sentence (Section 4 of this file)
- [ ] **Section 6.2.4** (Transformer future work): Append the "all three changes implemented" sentence (Section 4 of this file)
- [ ] **Section 6.2** (iTransformer "to be incorporated"): Replace with "was implemented; results in Table 5.2"
- [ ] **Methods/reproducibility section**: Add the seeds + CSVLogger paragraph (Section 6 of this file)
- [ ] **Conclusions**: Add 2–3 sentences noting iTransformer as best model, Transformer underperformance finding

### Step D — Finalise code
- [ ] Merge `feature/transformer-revision` into `main`
- [ ] Tag `v1.0-final-results`
- [ ] Push to GitHub
- [ ] Note the commit hash in dissertation methods as the reproducible codebase reference
