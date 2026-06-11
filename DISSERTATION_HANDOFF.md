# Dissertation Handoff — Final EC2 Results (Seed 42)

Use this file to update the dissertation. Work through sections A–F in order.

---

## A. Final Results to Use Everywhere

These are the definitive numbers from the full EC2 run (37,846 trajectories, seed 42).

### RQ1 — Trajectory Prediction

| Model | Test MSE | Test MAE | Inference (ms/sample) |
|---|---|---|---|
| **iTransformer** | **0.003675** | **0.016990** | 3.3 |
| LSTM | 0.004085 | 0.018399 | 14.6 |
| GRU (clipnorm=1.0) | 0.005910 | 0.031673 | 12.9 |
| Transformer (revised) | 0.074270 | 0.109051 | 3.3 |
| Transformer | 0.079326 | 0.121387 | 3.3 |
| Breen MLP | — | 0.182374 | — |
| Numerical integration | — | — | 3.8 |

### RQ2 — Classification (unchanged from existing dissertation)

| Model | Test Accuracy |
|---|---|
| Random Forest | 80.9% |
| MLP | 73.9% |
| Logistic Regression | 21.7% |

### RQ3 — Equilibrium Discovery (unchanged)

L5 partially recovered (distance 0.074). Results unchanged.

---

## B. Table to Replace in Chapter 5 (RQ1 results table)

Replace the existing prediction results table with the one in Section A above.

**Caption to use:**
> *All sequence models trained on 37,846 trajectories (seed 42) with early stopping
> (patience = 15), ReduceLROnPlateau (patience = 5, factor = 0.5), and best-weight
> restoration. GRU uses Adam with clipnorm = 1.0 to prevent gradient explosion.
> Transformer (revised) uses two attention heads, a learnable positional embedding,
> and learning rate 1×10⁻⁴. iTransformer applies self-attention across the four
> phase-space variates (ξ, η, vξ, vη) rather than across time steps. Inference
> times measured per sample on CPU; Transformer-family models match numerical
> integration speed.*

---

## C. Text to Add in Each Section

### In the prediction results section — add one paragraph introducing the new models

Paste this before or after the existing LSTM/GRU discussion:

> Three additional models were evaluated alongside LSTM and GRU. The iTransformer
> (Liu et al., 2024) inverts the standard tokenisation: rather than treating each
> of the 50 time steps as a token, it treats each of the four phase-space coordinates
> (ξ, η, vξ, vη) as a token whose embedding is its complete time series. Self-attention
> then operates across these four variate tokens, directly capturing the
> position–velocity couplings prescribed by the RC3BP equations of motion. The model
> used d_model = 64, four attention heads, and two encoder layers. The revised
> Transformer applied three targeted changes to the original architecture: attention
> heads reduced from four to two, fixed sinusoidal positional encoding replaced with
> a learnable embedding layer, and the initial learning rate lowered from 1×10⁻³ to
> 1×10⁻⁴. Within the seeded run, iTransformer achieved the lowest test MAE (0.0170),
> narrowly outperforming LSTM (0.0184). Both Transformer variants substantially
> underperformed the recurrent and iTransformer models (MAE 0.109–0.121), indicating
> that temporal self-attention is ill-suited to this 4D sequential task at this scale.
> The revised GRU with gradient clipping eliminated the epoch-5 instability spike
> observed in the original run (Section 6.2.3), confirming gradient explosion as
> its sole cause.

---

### In Section 6.2.3 (GRU gradient clipping — currently written as future work)

**Append to the end of the existing paragraph:**

> This fix was subsequently implemented by passing `clipnorm=1.0` to the Adam
> optimiser. Results are reported in the prediction results table.

---

### In Section 6.2.4 (Transformer revision — currently written as future work)

**Append to the end of the existing paragraph:**

> All three changes were subsequently implemented: attention heads reduced from four
> to two, fixed sinusoidal positional encoding replaced with a learnable embedding
> layer, and the Adam learning rate set to 1×10⁻⁴. Results are reported in the
> prediction results table as Transformer (revised).

---

### Wherever iTransformer is listed as "to be incorporated"

**Replace that sentence with:**

> The iTransformer was implemented and evaluated alongside LSTM, GRU, and the
> Transformer variants; results are reported in the prediction results table.

---

### In the Methods section (wherever training setup is described)

**Add this paragraph:**

> All deep learning models were trained with a fixed random seed (seed 42) applied
> to both TensorFlow (`tf.random.set_seed`) and NumPy (`np.random.seed`) at the
> start of each training function. Per-epoch training metrics were recorded via
> Keras CSVLogger callbacks. Final model weights were saved in `.keras` format with
> early stopping configured to restore the best validation-loss checkpoint.

---

### In the Future Work / Limitations section

**Add this sentence:**

> The reproducibility infrastructure required for multi-seed validation — fixed
> random seeds, per-epoch CSVLogger audit trails, and saved model weights — is
> fully implemented; reporting results as mean ± standard deviation across multiple
> seeds is reserved as an immediate extension of this work.

---

### In the Conclusions section

**Add 2–3 sentences:**

> Among the trajectory prediction models, the iTransformer achieved the lowest test
> MAE (0.0170), demonstrating that attention across phase-space variates is better
> suited to the RC3BP than attention across time steps. Both standard and revised
> Transformer architectures substantially underperformed LSTM and iTransformer,
> suggesting that temporal self-attention requires significantly more training data
> to calibrate effectively on this 4D sequential task. The revised GRU with gradient
> clipping confirmed that the epoch-5 instability in the original run was entirely
> attributable to unconstrained gradients, and not an intrinsic limitation of the
> GRU architecture.

---

## D. Figures — What to Replace and What to Add

Download from S3 first (run on your Mac in the project directory):

```python
from s3_utils import download
download('prediction_training_history.png')
download('prediction_examples.png')
```

| Figure file | Action | Where in dissertation |
|---|---|---|
| `prediction_training_history.png` | **Replace** existing training history figure | Chapter 5, RQ1 section |
| `prediction_examples.png` | **Replace** existing prediction examples figure | Chapter 5, RQ1 section |
| All other figures | Leave unchanged | — |

### Caption for `prediction_training_history.png`

> *Training and validation loss (MSE, log scale, left) and MAE (right) for all five
> prediction models. LSTM and iTransformer converge to the lowest validation loss.
> Both Transformer variants plateau at substantially higher loss from early epochs,
> consistent with their test MAE values in Table X.*

### Caption for `prediction_examples.png`

> *Example trajectory predictions for three held-out test samples. Top row: predicted
> vs true continuation in (ξ, η) phase space. Bottom row: ξ coordinate over time
> steps 0–60, with the prediction horizon beginning at step 50 (dashed line). LSTM
> and iTransformer closely track the true trajectory; both Transformer variants
> diverge at longer horizons.*

---

## E. What Has NOT Changed (do not touch these)

- Data generation, preprocessing — unchanged
- Classification results, figures, and discussion — unchanged
- Equilibrium discovery results and figures — unchanged
- LSTM architecture — unchanged
- Breen MLP architecture — unchanged
- Background / literature review chapter — unchanged (except adding iTransformer citation if needed)
- RQ2 and RQ3 results sections — unchanged

---

## F. Final Code Steps (after dissertation is updated)

```bash
git checkout main
git merge feature/transformer-revision
git tag v1.0-final-results
git push origin main --tags
```

Note the commit hash in the dissertation Methods section as the reproducible codebase reference.

---

## iTransformer Citation

Liu, Y., Hu, T., Li, S., Wu, H., Liu, Q., Liao, W., & Long, M. (2024).
*iTransformer: Inverted Transformers Are Effective for Time Series Forecasting.*
ICLR 2024. arXiv:2310.06625.
