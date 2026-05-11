# Results Discussion

## Methodology — Data Leakage Fix

A critical data leakage issue was identified and fixed in `prepare_prediction_data()`. The original code created sliding-window sequences across all trajectories first, then split the resulting sequences into train/val/test. This allowed windows from the same trajectory to appear in multiple splits — a sequence starting at timestep 0 of a trajectory could end up in train while a sequence starting at timestep 1 of the same trajectory ended up in test. The model was effectively being tested on data it had already seen in a slightly shifted form, artificially deflating reported MAE.

The fix splits trajectories first (70/15/15) and creates sequences independently within each split. This guarantees no trajectory contributes to more than one set. A dedicated test (`test_no_data_leakage()`) was added to `test_prediction.py` to verify this on every run.

---

## RQ2 — Stability Classification

### Class Imbalance Drives Accuracy

The dataset is heavily skewed toward Escape trajectories (~73% of samples). This means a naive classifier that predicts Escape for every input would achieve ~73% accuracy without learning anything meaningful. The reported test accuracies of ~78% are only marginally better than this baseline, so headline accuracy alone is a misleading success metric here.

A more honest evaluation uses the macro-averaged F1-score. The Random Forest achieves a macro F1 of 0.44 — well below its weighted F1 of 0.77 — revealing that strong Escape performance masks poor generalisation across the other classes.

### Chaotic Class is Unlearnable

The Chaotic class has only ~5 samples in the entire dataset (~0.1%), leaving just 1 sample in the test set. No model correctly classifies it. This is not a model failure — it is a data scarcity problem. Meaningful classification of chaotic trajectories would require a significantly larger and more balanced dataset, or targeted oversampling techniques (e.g. SMOTE).

### Logistic Regression Collapses to Majority Class

Logistic Regression predicts everything as Escape, achieving 0 correct predictions for Stable, Chaotic, and Collision. This is a textbook symptom of a linear model overwhelmed by class imbalance. It serves as a useful lower bound but offers no real discriminative power across trajectory types.

### Random Forest Overfits but Still Generalises

Random Forest achieves 100% training accuracy (perfect memorisation of the training set) yet drops to ~78.9% on the test set. The ~21% generalisation gap indicates significant overfitting. Despite this, it remains the best-performing model and does learn partial structure for Stable and Collision classes. This suggests the decision boundaries for these classes exist in the feature space but are not smooth — tree ensembles can find them through memorisation, whereas a regularised model like MLP cannot as easily.

### MLP and Random Forest Converge at Different Failure Points

Both RF and MLP achieve ~78% test accuracy but with different confusion patterns. RF is better at Stable (31 correct vs. 7 for MLP) while MLP is better at Collision (48 correct vs. 37 for RF). This suggests the two models learn complementary features — an ensemble of RF + MLP could potentially outperform either individually.

### Weighted Accuracy Overstates Performance

The weighted F1 of 0.77 is dominated by the Escape class (F1 = 0.90, support = 520). The minority classes — which are arguably more scientifically interesting (Stable: F1 = 0.45, Collision: F1 = 0.40) — are underserved by all models. Future work should consider class-weighted loss functions or resampling strategies to improve minority class performance.

### Comparison to Paper Expectations

The README states expected accuracy of 80–85% for Random Forest and 82–87% for MLP. The achieved values (~78.9% and ~78.7%) fall below these ranges. This is consistent with the expanded 4-class problem (vs. a simpler binary stable/unstable framing) and the severe class imbalance introduced by the Chaotic class having near-zero representation.

---

## RQ1 — Trajectory Prediction

### Why Sequence-Based Input

Trajectory prediction is inherently a temporal problem. To predict where a body will be at timestep 60, the model needs to know where it was at timesteps 1–50 — history matters. A single snapshot of position and velocity is insufficient because the three-body problem is chaotic: tiny differences in past history lead to wildly different futures. This rules out models like Logistic Regression or standard MLP, which treat each input independently with no notion of temporal order.

### Why LSTM and GRU

LSTM and GRU were designed specifically for ordered sequences where what happened earlier affects what comes later. A standard neural network has no memory — if fed timestep 1 then timestep 2, it treats them as unrelated inputs. LSTM and GRU solve this by maintaining a hidden state, a compressed memory updated at each timestep, that carries information forward through the sequence.

LSTM uses three gates (forget, input, output) to control what information to keep, discard, or pass forward. This is particularly relevant for three-body trajectories where a near-collision 30 timesteps ago still influences current motion. GRU is a simplified version with two gates and fewer parameters — it trains faster and often achieves comparable accuracy, making it a natural comparison point.

The non-linearity and sensitivity to initial conditions in the three-body problem mean linear models cannot capture the dynamics. Sequence structure combined with non-linearity makes LSTM and GRU the natural baseline architectures for this task.

### Sliding Window Stride and Overfitting

The sliding window approach creates training sequences by stepping through each trajectory. With stride=1, consecutive windows share 49 of 50 timesteps — a 98% overlap — producing ~2.1 million near-duplicate sequences. This is a direct cause of overfitting: the model is exposed to the same trajectory patterns hundreds of times within a single epoch, leading it to memorise specific training trajectories rather than learn general dynamics.

Increasing the stride to 5 reduces sequences to ~420k while maintaining 90% overlap between consecutive windows. This removes redundancy without discarding trajectory coverage, reducing overfitting risk and cutting training time by ~5x. Stride is therefore both a computational and a regularisation hyperparameter — a trade-off worth documenting explicitly.

### Why Transformer for Comparison

The Transformer approaches sequence modelling differently from LSTM/GRU. Rather than processing timesteps one-by-one with a running hidden state, it uses self-attention to look at all 50 input timesteps simultaneously and learn which are most relevant to each prediction. This is meaningful for the three-body problem because a body's position at timestep 10 may be more predictively relevant to timestep 55 than timestep 49 is — a relationship LSTM can struggle to maintain over long sequences. The Transformer can in principle capture these long-range dependencies directly.

### LSTM Results and Training Behaviour

The LSTM converged at epoch 54 (out of 100), stopped early by EarlyStopping after val_loss failed to improve for 15 consecutive epochs. Final results:

- **Test MSE**: 0.001140
- **Test MAE**: 0.007819 (normalised units)

The train/val loss gap was small (train MSE 0.000954 vs val MSE 0.0012), indicating good generalisation — a direct improvement attributable to the stride=5 fix reducing near-duplicate sequences.

A notable pattern: training MAE (0.0118) was higher than validation MAE (0.0081). This is expected behaviour with Dropout — the regularisation layer randomly disables 20% of neurons during training but is fully active during validation and test, making training metrics appear worse than they are. This is not a sign of underfitting.

The learning rate trace also tells the convergence story. Adam's default learning rate of 0.001 was halved approximately 8 times by ReduceLROnPlateau, reaching 3.9e-06 by epoch 54 — at which point the model was taking negligibly small steps and had effectively converged, triggering early stopping.

### GRU Results and Model Comparison

The GRU converged at epoch ~71, slightly later than LSTM but achieving marginally better results:

| Model | Test MSE | Test MAE | Inference Time |
|-------|----------|----------|----------------|
| LSTM  | 0.001140 | 0.008072 | 16.15 ms/sample |
| GRU   | 0.001106 | 0.007398 | 22.21 ms/sample |

GRU achieves lower MSE and MAE despite having fewer parameters — consistent with the general finding that GRU's simpler gating mechanism is sufficient for many sequence tasks and can generalise slightly better than LSTM when data is limited.

### Neural Networks Are Slower Than Numerical Integration on CPU

A critical and counterintuitive finding: on CPU, both models are significantly **slower** than numerical integration (LSTM: 16.15ms, GRU: 22.21ms vs numerical: 3.39ms). This directly contradicts the speedup claim in the paper's motivation.

The explanation is straightforward: TensorFlow on CPU carries significant inference overhead — graph execution, memory allocation, and data movement — that dominates for small batch sizes. Numerical integration of just 10 timesteps is a lightweight ODE solve that runs efficiently in native code.

The speedup claim holds on GPU, where neural network inference is highly parallelised and the overhead amortises across large batches. This makes EC2 GPU deployment not just a convenience but a **scientific necessity** for the efficiency argument in RQ1 to hold. This distinction is worth making explicitly in the dissertation.

---

## RQ3 — Equilibrium Discovery

### Only One Lagrange Point Partially Recovered

The unsupervised clustering found 4 regions of low-velocity points across the dataset:

| Cluster | Center | Size | Nearest Known Point | Distance |
|---------|--------|------|---------------------|----------|
| 0 | (-0.718, -0.287) | 17 | L5 | 1.085 |
| 1 | (0.078, 0.041) | 48 | L4 | 0.834 |
| 2 | (-0.543, 0.326) | 17 | L4 | 0.918 |
| 3 | (0.018, 0.907) | 11 | L4 | **0.187** |

Only Cluster 3 is meaningfully close to a known Lagrange point (L4 at (0.2, 0.866), distance 0.187). The other three clusters are likely near-collision or momentary low-velocity events unrelated to true equilibria. L5 was not recovered at all.

This is a weak result for RQ3. The low-velocity heuristic is an imprecise proxy for equilibrium — a body can have near-zero velocity at any turning point in a trajectory, not just near Lagrange points. A more principled approach would filter by the Jacobi integral value or use the equations of motion directly to identify equilibrium candidates.
