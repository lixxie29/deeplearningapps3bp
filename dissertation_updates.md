# Dissertation Updates

Everything you need to update the dissertation and run the final experiments.
Read top-to-bottom the first time; use as a checklist afterwards.

---

## 1. What changed in the code (summary)

| File | What changed |
|---|---|
| `models.py` | GRU optimizer: `'adam'` → `Adam(clipnorm=1.0)` |
| `models.py` | iTransformer added as new model (`build_itransformer_predictor`) |
| `models.py` | Transformer: pending revision on `feature/transformer-revision` branch |
| `train_prediction.py` | Seeds added (`seed=42`); `--smoke` and `--seed` CLI flags |
| `train_breen_baseline.py` | Seeds added; CSVLogger added; `--smoke` and `--seed` CLI flags |
| `s3_utils.py` | Added `.keras` model files and `_training_log.csv` files to upload list |

Nothing else changed. All models still have the same architecture except GRU (one optimizer line) and the pending Transformer revision.

---

## 2. The Transformer revision (still pending)

Still to be implemented on `feature/transformer-revision`. Three changes to `build_transformer_predictor`:

1. `num_heads=4` → `num_heads=2` (less overparameterised for 4D input)
2. Fixed sinusoidal positional encoding → learnable `Embedding` layer
3. `optimizer='adam'` → `Adam(learning_rate=1e-4)` (default 1e-3 suspected of overshooting)

**Say the word and these get applied.** Nothing else in the codebase changes.

---

## 3. How to run

### Step 1 — Local smoke test (run this now, takes ~2 minutes)

Verifies the code runs without errors. Do this before touching EC2.

```bash
python train_prediction.py --smoke
python train_breen_baseline.py --smoke
```

If both complete without crashing, the code is good.

### Step 2 — EC2 real run (produces final dissertation results)

Run this once per seed. Three seeds = three runs = mean ± std in Table 5.2.

```bash
# Seed 1 (primary run)
python train_prediction.py --seed 42
python train_breen_baseline.py --seed 42

# Seed 2
python train_prediction.py --seed 123
python train_breen_baseline.py --seed 123

# Seed 3
python train_prediction.py --seed 456
python train_breen_baseline.py --seed 456
```

You do **not** re-run classification or equilibrium discovery — those models haven't changed.

After all three runs, upload to S3:
```bash
python -c "from s3_utils import upload_all_results; upload_all_results()"
```

### What gets produced per run

- `lstm_model.keras`, `gru_model.keras`, `transformer_model.keras`, `itransformer_model.keras`, `breen_model.keras`
- `lstm_training_log.csv`, `gru_training_log.csv`, `transformer_training_log.csv`, `itransformer_training_log.csv`, `breen_training_log.csv`
- `prediction_results.pkl`, `breen_results.pkl`
- Training history and example prediction plots

---

## 4. Dissertation changes — section by section

### Table 5.2 (main results table for RQ1)

Add these rows. Use the numbers from the three EC2 runs to compute mean ± std.

| Model | Test MAE | Notes |
|---|---|---|
| LSTM | *(existing)* | Unchanged |
| GRU | *(existing)* | Original run, epoch-5 spike observed |
| **GRU (clipnorm=1.0)** | *(new EC2 result)* | Gradient clipping applied |
| Transformer | *(existing)* | Original architecture, flat loss |
| **Transformer (revised)** | *(new EC2 result)* | 2 heads, learnable PE, LR=1e-4 |
| **iTransformer** | *(new EC2 result)* | New model, variate-attention |
| Breen MLP | *(existing or updated)* | |

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
> the epoch-5 validation loss spike and produced [X% lower / comparable] test MAE,
> confirming gradient explosion as the sole cause of the original instability.

**Add after the Transformer discussion** — one sentence:

> The revised Transformer (Table 5.2, Transformer revised) with reduced heads,
> learnable positional embeddings, and a lower learning rate [improved upon /
> still underperformed] the original, [suggesting the flat loss was a learning
> rate issue / confirming the architecture is not suited to this task at this scale].

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

## 5. Reproducibility — what to state in the methods section

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

## 6. Branch strategy

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

## 7. What is NOT changing

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
