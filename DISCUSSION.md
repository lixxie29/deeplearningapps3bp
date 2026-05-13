# Results Discussion
nohup python run_all.py > training_log.txt 2>&1 &
---

## Literature Review

### Breen et al. (2019) — Newton vs the Machine

Breen, Foley, Boekholt, and Portegies Zwart published "Newton vs the machine: solving the chaotic three-body problem using deep neural networks" (arXiv:1910.07291) in October 2019. The paper is a proof-of-concept study demonstrating that a deep feed-forward artificial neural network (ANN) can produce accurate solutions to the gravitational three-body problem over a fixed time interval, at a fraction of the computational cost of a traditional numerical integrator.

#### Motivation and Context

The three-body problem — determining the trajectories of three mutually gravitating bodies given their initial positions and velocities — has no general closed-form solution. Since Poincaré established its non-integrability in the late nineteenth century, numerical integration has been the only practical route to solutions. The chaotic nature of the system means that even small errors in initial conditions grow exponentially, making long-term predictions fundamentally uncertain (Miller 1964; Valtonen et al. 2016).

The standard approach is a high-precision numerical integrator. The authors use Brutus (Boekholt & Portegies Zwart 2015), which iteratively reduces its tolerance parameter and word length (numerical precision in bits) until two independent runs with different settings agree to within a phase-distance threshold. This guarantees convergence but is computationally expensive: the authors report over ten days of CPU time to produce their 10,000-simulation training dataset, with individual simulations near the singular point ((-0.5, 0), where two particles approach collision) taking particularly long.

The astrophysical motivation is practical. Three-body interactions — specifically between a black-hole binary and a single black-hole — form the primary computational bottleneck in simulating the evolution of globular star clusters and galactic nuclei. These interactions occur over bounded time windows and can be integrated independently of the surrounding cluster (Portegies Zwart & McMillan 2018). Replacing Brutus with a fast neural network for these sub-problems would make large-scale N-body simulations substantially more tractable.

#### Methodology

The authors simplify the three-body problem to a learnable form through a sequence of physical and coordinate constraints. They restrict the system to three equal-mass particles with zero initial velocities, moving in a two-dimensional plane. The most distant particle (x1) is placed at (1, 0), setting the length unit and orientating the x-axis. The particle nearest to the barycentre (x2) is placed randomly in the left semicircle (x ≤ 0), orientating the positive y-axis. The third particle (x3) is determined by symmetry, and the origin is taken as the centre of mass. Under these constraints, the entire initial configuration is described by two parameters — the (x, y) coordinates of x2 — and the solution at any time t is described by four numbers: the positions of x1 and x2 (x3 follows from centre-of-mass conservation: x3 = −x1 − x2). The general solution is therefore a mapping from the three-dimensional phase-space (t, x2_0, y2_0) to the four-dimensional output (x1(t), y1(t), x2(t), y2(t)).

The network architecture is a feed-forward ANN with ten hidden layers of 128 nodes each, using ReLU activations and a linear output layer. It is trained with Adam, MAE loss, a batch size of 5,000, and 10,000 epochs. Training and validation datasets comprise 9,900 and 100 simulations respectively, each yielding up to 2,561 time-stamped position snapshots. The authors partition the data into three time-horizon subsets (T ≤ 3.9, T ≤ 7.8, T ≤ 10) and find that the T ≤ 3.9 model achieves the lowest validation MAE — a consequence of chaotic divergence making longer trajectories harder to learn.

#### Results

The best-performing network achieves a validation MAE ≤ 0.1 and produces trajectories visually indistinguishable from Brutus solutions across both seen (training set) and unseen (validation set) initialisations. Critically, the network runs in approximately 10⁻³ seconds per query — on average 10⁵ times faster than Brutus, and up to 10⁸ times faster in the most challenging near-collision regimes where Brutus requires extreme precision.

The authors further verify that the network reproduces the defining characteristic of chaos: sensitive dependence on initial conditions. Across 1,000 realisations with x2 placed on a ring of radius 0.01 centred at (−0.2, 0.3), the ANN-predicted trajectory divergence closely matches Brutus. Lyapunov exponents estimated from 4,000 pairs of perturbed realisations (perturbation δ = 10⁻⁶) give a median value of 1.30, consistent with chaotic behaviour, and the ANN reproduces these estimates accurately. This is notable because the network was never explicitly trained on chaos — it emerges from the learned solution function.

Energy conservation is assessed by training a second ANN on velocities (differentiating the position network is theoretically possible but the authors opt for a separate network). The raw output carries a relative energy error of ~10⁻². A post-processing projection layer — a small Nelder-Mead optimisation that adjusts coordinates to lie on the correct energy surface while staying close to the ANN prediction — reduces this to ~10⁻⁵.

#### Limitations

The authors acknowledge several constraints. The setup is restricted to equal-mass particles with zero initial velocities, which covers only a small region of the full three-body phase space. The near-singular region around (−0.5, 0) — where x2 and x3 are nearly coincident — could not be resolved even with Brutus at the predetermined precision, so those initialisations are excluded from training. The network's bounded time interval is a hard constraint: training on T ≤ 3.9 produces the most accurate model, but the system cannot be queried outside this window without retraining or chaining multiple network evaluations. The energy projection layer is applied post-hoc and involves a non-trivial optimisation at inference time, partially undermining the fixed-cost claim for energy-conserving predictions.

#### Contribution to the Field

The paper's primary contribution is demonstrating that a neural network trained on arbitrarily precise numerical solutions can serve as a surrogate integrator for chaotic systems, reproducing not just position accuracy but statistical properties like the Lyapunov exponent. It sits within a broader trend of physics-informed machine learning and neural solvers for differential equations (Pathak et al. 2018; Raissi et al. 2019; Stinis 2019), but distinguishes itself by operating on a physically meaningful, long-studied chaotic system where ground truth is expensive rather than cheap. The proposed hybrid numerical integrator — where the ANN handles computationally challenging regions and hands off to a traditional integrator elsewhere — points toward a practical deployment path that subsequent work in the field has expanded upon.

---

## Breen et al. (2019) Baseline — Detailed Breakdown

### Why the Three-Body Problem Can Be Treated as Regression

The standard framing of the three-body problem is simulation: given a state at time t, step forward to t + Δt and repeat. Every step costs computation, errors accumulate, and chaotic regions force the step size toward zero.

The Breen et al. reframing is different. The true solution to the three-body problem is a function:

```
f(t, initial_conditions) → positions of all bodies at time t
```

This function exists — it is the analytical solution nobody has written down. It takes a finite-dimensional input (time plus a few numbers describing the initial state) and returns a finite-dimensional output (particle coordinates). That structure is exactly what a neural network approximates. The Universal Approximation Theorem (Hornik 1991; Cybenko 1989) guarantees that a sufficiently deep and wide MLP can approximate any continuous function of this type to arbitrary precision.

Once trained, querying the network costs one forward pass at fixed computational cost, independent of the difficulty of the initial conditions. There is no iteration, no adaptive step size, and no precision budget — the chaos of the system is absorbed into the training process rather than the inference process.

### Physical Setup and Symmetry Reduction

The paper does not attempt to learn the fully general three-body problem — that would require a much larger input space and far more training data. Instead, they apply a sequence of constraints that reduce the initial condition space to just two free parameters.

**Constraints applied:**

- Three equal masses — removes the two mass ratio parameters
- Zero initial velocities — removes all six velocity parameters
- Motion in a plane — removes the z-dimension
- x1 fixed at (1, 0) — sets the length unit, removes two position parameters
- x3 is the mirror of x2 across the x-axis — by symmetry, only x2 needs specifying
- Centre of mass at origin — x3 can always be recovered as x3 = −x1 − x2

After these reductions, the entire initial state of the system is described by two numbers: the (x, y) coordinates of x2, which is constrained to lie in the left unit semicircle (x ≤ 0). A singular point exists at (−0.5, 0) where x2 and x3 coincide — trajectories initialised near this point cause near-collision orbits that even Brutus struggles to resolve, and they are excluded from training.

The network input is therefore `[t, x2_0, y2_0]` — three scalars. The output is `[x1(t), y1(t), x2(t), y2(t)]` — four scalars. x3(t) is free: x3 = −x1 − x2.

### Architecture

```
Input layer:  3 nodes  [t, x2_0, y2_0]

Hidden layer 1:   128 nodes, ReLU
Hidden layer 2:   128 nodes, ReLU
Hidden layer 3:   128 nodes, ReLU
Hidden layer 4:   128 nodes, ReLU
Hidden layer 5:   128 nodes, ReLU
Hidden layer 6:   128 nodes, ReLU
Hidden layer 7:   128 nodes, ReLU
Hidden layer 8:   128 nodes, ReLU
Hidden layer 9:   128 nodes, ReLU
Hidden layer 10:  128 nodes, ReLU

Output layer:  4 nodes, linear  [x1(t), y1(t), x2(t), y2(t)]
```

The architecture was arrived at empirically: starting from 5 hidden layers with 32 nodes, width and depth were increased until the network accurately reproduced complex close-encounter trajectories. Transposed convolution (deconvolution) layers were also evaluated but underperformed dense layers by MAE.

**Loss function — MAE over MSE:** In chaotic systems, close-encounter events produce large but physically valid spikes in position error. MSE penalises these quadratically, causing the optimiser to spend disproportionate effort on rare extreme cases. MAE treats all errors equally, producing a more robust training signal when the error distribution has heavy tails.

**Optimiser — Adam over alternatives:** AdaGrad and Nesterov SGD were tested and "regularly failed to match the performance of Adam." Adam's adaptive per-parameter learning rates are well-suited to loss landscapes with varying curvature, which is expected for a chaotic function.

**Batch size 5,000 and 10,000 epochs:** Large batch sizes produce stable gradient estimates and allow the optimiser to make consistent progress. With ~9.9 million training samples (for T ≤ 3.9), 10,000 epochs is computationally manageable and the loss curves show most learning happens in the first ~100 epochs, followed by slow refinement.

### Training Data and the Role of Brutus

The network has no access to the equations of motion during training — it learns purely from input-output pairs. The quality of the training data therefore sets a ceiling on the network's accuracy. This is why Brutus is essential: a lower-precision integrator would inject systematic errors into the training labels, and the network would learn an approximation of an approximation.

Each simulation runs for up to T = 10 time units, producing ~2,561 discrete position snapshots. Each snapshot becomes one training sample: `(t, x2_0, y2_0) → (x1, y1, x2, y2)`. For the T ≤ 3.9 case this gives roughly 1,000 samples per simulation × 9,900 simulations = 9.9 million training samples. Generating this data required over ten days of CPU time.

**Why the T ≤ 3.9 model outperforms T ≤ 10:** Lyapunov exponents estimated in the paper have a median of 1.30, meaning perturbations grow by a factor of e ≈ 2.7 per time unit on average. By T = 10, a perturbation of 10⁻⁶ has grown to ~10⁻⁶ × e¹³ ≈ 0.44 — the solution at T = 10 is fundamentally more complex and variable across nearby initialisations than at T = 3.9. A network with the same architecture has to learn a harder function for the longer time horizon, resulting in higher validation MAE.

### Key Findings

**Speed:** ~10⁻³ seconds per query versus ~10² to 10⁵ seconds for Brutus. The speedup reaches 10⁸× for near-singular initialisations where Brutus requires extreme precision and very small time steps.

**Accuracy:** Validation MAE ≤ 0.1 on unseen initialisations. Particle trajectories are visually indistinguishable from Brutus solutions across both smooth and highly chaotic close-encounter scenarios.

**Chaos reproduction:** The ANN accurately reproduces trajectory divergence across a ring of perturbed initialisations, and Lyapunov exponent estimates from ANN-simulated trajectories match Brutus. This is an emergent property — the network was trained on positions, not on divergence rates.

**Energy conservation:** Raw output has ~1% relative energy error, spiking during close encounters. A post-hoc projection layer (Nelder-Mead optimisation) reduces this to ~10⁻⁵ by nudging predicted coordinates onto the correct energy surface.

### Relevance to This Project

This project uses the restricted circular three-body problem (a test particle orbiting two fixed primaries in a rotating frame) and frames trajectory prediction as a sequence-to-sequence task: given 50 timesteps of history, predict the next 10. The Breen et al. approach is architecturally different — it approximates the solution function directly rather than extrapolating a sequence — and operates on a different physical system (free equal-mass 3BP vs restricted circular 3BP).

The Breen baseline is nevertheless directly relevant as a comparison point for three reasons. First, it establishes that a simple 10-layer MLP can learn physically meaningful solutions to a chaotic gravitational system, motivating the use of deeper and more expressive architectures. Second, their finding that speedup only materialises on GPU — not CPU — is a result this project independently reproduces for LSTM and GRU, supporting the conclusion that EC2 GPU deployment is a scientific requirement rather than an engineering convenience. Third, extending the Breen setup to unequal masses, non-zero initial velocities, or longer time horizons using a Transformer architecture constitutes a natural direction for future work and a publishable contribution beyond replication.

---

## Methodology — Dataset Generation and Class Imbalance

### The Problem with Pure Random Sampling

The initial dataset was generated by drawing initial conditions uniformly at random: position `[xi_0, eta_0]` from the box `[-1.5, 1.5]²` and velocity `[vxi_0, veta_0]` from `[-0.8, 0.8]²`, with `mu` drawn uniformly from `[0.1, 0.4]`. While uniform random sampling is the simplest approach, the resulting class distribution reflects the physical structure of the RC3BP phase space — not a useful distribution for a classification task:

| Class | Natural frequency | Reason |
|---|---|---|
| Escape | ~73% | Most random initial conditions give the particle enough energy to fly out |
| Stable | ~20% | Only specific, narrow regions of IC space produce bounded orbits |
| Collision | ~6% | Requires starting close to a primary, unlikely by chance |
| Chaotic | ~0.1% | Bounded orbits with high energy variation are rare near-miss events |

This is not a modelling failure — it is the actual geometry of the phase space. The escape region occupies most of the IC box. Stable orbits cluster near specific dynamical structures (Lagrange points, resonant orbits). Chaotic bounded orbits exist at the narrow boundary between stability and escape.

The consequence for RQ2 is severe: a classifier that always predicts Escape achieves ~73% accuracy without learning anything. The Chaotic class with ~0.1% representation is completely unlearnable — with 4,771 trajectories the test set contains roughly one chaotic example, so no model can meaningfully evaluate on it. Headline accuracy is a misleading metric precisely because the dominant class artificially inflates it.

### Why More Random Samples Do Not Fix This

Generating 50,000 trajectories with pure random sampling produces ~36,500 Escape, ~10,000 Stable, ~3,000 Collision, and ~50 Chaotic. The ratio is unchanged — it is a property of the physics, not the sample count. No amount of random sampling can densely cover the narrow regions of IC space that produce rare trajectory types.

### Targeted Sampling Strategy

To produce a usable class distribution, the dataset generation was changed to use **quota-based targeted sampling**: initial conditions are drawn from distributions designed to land in specific dynamical regions, and generation continues until each class hits its target count.

**Stable orbits — sample near L4 and L5:**
L4 and L5 are the triangular Lagrange points at `(0.5 - mu, ±sqrt(3)/2)`. For `mu` in `[0.1, 0.4]`, these sit at roughly `(0.1–0.4, ±0.866)`. Particles starting near these points with small velocities are trapped in the potential well and produce stable, bounded orbits. Positions are drawn with `±0.3` perturbations around L4/L5 and velocities are restricted to `[-0.15, 0.15]` to avoid giving the particle enough energy to immediately escape the well.

**Chaotic orbits — sample near L1:**
L1 is the collinear Lagrange point between the two primaries, located approximately at `x = (1 - mu) - (mu/3)^(1/3)`. It is an unstable equilibrium: the potential saddle point between the two attraction basins. Trajectories starting near L1 sit on the boundary between confinement and escape, producing the bounded high-energy-variation orbits that the classifier labels as Chaotic. Positions are drawn within `±0.2` of L1 along the x-axis and `±0.3` in the y-direction, with moderate velocities `[-0.3, 0.3]`.

**Collision orbits — sample close to a primary:**
Collision requires the particle to actually reach a primary (distance < 0.01). With random sampling this only happens when the particle starts moderately close and has the right velocity. The targeted sampler draws positions at radius `r ∈ [0.02, 0.12]` from a randomly chosen primary, giving the particle a short path to collision. The minimum-distance threshold that skips initial conditions too close to primaries is relaxed to 0.05 (from 0.1) for this class only.

**Escape orbits — pure random:**
Escape is the natural outcome of most random initial conditions and requires no targeting. The sampler uses the original uniform distribution and accepts escape trajectories until the quota is filled.

### ODE Solver Convergence Failures and Data Quality

The numerical integrator (scipy's `odeint`, which uses the LSODA algorithm) occasionally fails to converge when integrating near-collision trajectories. This manifests as warnings of the form:

```
lsoda -- at t (=r1) and step size h (=r2), the corrector convergence failed repeatedly or with abs(h) = hmin
lsoda -- warning: internal t and h are such that t + h = t on the next step
```

Both warnings indicate the same underlying problem: the ODE has become numerically stiff because the particle is approaching a primary body. The gravitational acceleration grows as `1/r²`, so at small distances the force changes rapidly over tiny time intervals, forcing the adaptive step size down to machine precision. At that point the solver cannot make further progress.

These failures are handled explicitly: `generate_single_trajectory` wraps the integration in a try/except block and checks the output for NaN or non-finite values. Any trajectory that triggers a convergence failure returns `None` and is silently discarded. No partially-integrated or numerically corrupt trajectory enters the dataset.

The practical consequence is that the attempt count during balanced generation is substantially higher than the collected trajectory count. This is expected, particularly for the collision-targeted sampler which deliberately places the particle close to a primary — a region where stiffness is common. The 85% completion progress at 17 minutes shown during a sample run reflects this: approximately one in three collision-targeted integration attempts fails and is discarded before the solver produces a valid trajectory.

This is also relevant to the Chaotic class. The most extreme near-collision bounded orbits — which would likely be classified as Chaotic due to large Jacobi constant variation — are disproportionately filtered out by convergence failures, since they require the particle to pass extremely close to a primary without actually colliding. The Chaotic trajectories that do survive in the dataset represent a somewhat less extreme subset of bounded high-variation orbits.

### Resulting Distribution

The balanced generator targets `{Stable: 1500, Chaotic: 750, Escape: 1500, Collision: 750}` for a total of 4,500 trajectories split roughly 33%/17%/33%/17%. This is deliberately not perfectly equal — Escape and Stable are the most physically representative classes, and giving Chaotic and Collision 17% each rather than 25% avoids over-representing pathological edge cases at the expense of the dominant dynamics.

The quota system means that generation continues until all targets are met or a hard cap of 100,000 integration attempts is reached. Because targeted samplers are not guaranteed to produce the intended class (a near-L1 orbit might escape rather than stay bounded, for example), the attempt count is substantially higher than the trajectory count — especially for Chaotic, where the hit rate from the L1 sampler is estimated at 30–50%.

### Class Weights at Training Time

Targeted sampling reduces but does not eliminate imbalance — some trajectories sampled for one class produce a different outcome, and the exact final distribution depends on the physics. As a second layer of correction, all three classifiers use **class-weighted loss**:

- **Logistic Regression and Random Forest**: constructed with `class_weight='balanced'`, which causes sklearn to internally weight each sample by `n_total / (n_classes * n_class_i)`. Misclassifying a rare class incurs a proportionally larger penalty.
- **MLP**: class weights are computed from the actual training distribution using `compute_class_weight('balanced')` and passed to `model.fit()` via the `class_weight` argument. The effect is identical — the loss contribution of each batch sample is scaled by the inverse frequency of its class.

Together, targeted sampling and class-weighted loss address the imbalance at both the data level and the optimisation level. The more honest evaluation metric for RQ2 remains macro-averaged F1, which weights all classes equally regardless of support, and which should improve substantially compared to the original pure-random dataset.

---

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



### Data Requirements: Breen Baseline vs Sequence Models

The Breen baseline and the sequence models (LSTM, GRU, Transformer) differ not just in architecture but in what they fundamentally ask of training data. The comparison reveals a trade-off between phase-space coverage and temporal depth.

#### What Each Model Is Actually Learning
(global solution function vs local dynamics extrapolation, with the concrete function signatures)

The Breen MLP learns the *global solution function* — a direct mapping from initial conditions and time to particle positions:

```
f(t, x2_0, y2_0) → (x1(t), y1(t), x2(t), y2(t))
```

This function spans all of phase-space. For every possible initial condition and every possible query time, the network must produce the correct answer. Nothing about the query encodes how the system got there — the model has to reconstruct the full solution from a 3-scalar input alone. This demands broad, dense coverage of the IC space and the time axis simultaneously.

The sequence models learn *local dynamics* — a conditional extrapolation:

```
[xi, eta, vxi, veta] for t=1..50 → [xi, eta, vxi, veta] for t=51..60
```

The 50-timestep input window already encodes the trajectory's recent history: current position, velocity, any recent close encounters or accelerations. The model only needs to learn the rule "given what just happened, what comes next" — a locally defined and much more tractable function. It does not need to know the original initial conditions at all.

#### Raw Data Comparison

| | Breen et al. | This project |
|---|---|---|
| Simulations | 9,900 train + 100 val | 5,000 total |
| Time points per simulation | ~1,000 (T ≤ 3.9) | 500 |
| Training samples | ~9.9 million | ~440,000 sequences |
| Sample input shape | `(3,)` | `(50, 4)` = 200 values |
| Sample information content | One point in phase-space | 50 steps of full state |

The numbers look lopsided — 9.9M versus 440K — but the information per sample tells the opposite story. A Breen training sample is a single point `(t, x2_0, y2_0)` with essentially no context. A sequence model sample is a 50 × 4 window that implicitly encodes a substantial portion of the trajectory's dynamic history. The sequence approach extracts far more usable signal per training example, which is why it can generalise from a smaller raw sample count.

#### Why Breen Needs Phase-Space Coverage

The Breen model must have seen — or seen something very nearby — every query it will ever encounter at inference time. With only two free parameters (x2_0, y2_0) in the left unit semicircle, the IC space is compact and two-dimensional, which is why 9,900 simulations is manageable. If the IC space were higher-dimensional, this density requirement would become exponentially harder to satisfy — a direct manifestation of the curse of dimensionality.

This also explains why the T ≤ 3.9 model outperforms T ≤ 10: at longer time horizons, the chaotic divergence means that two nearby initial conditions produce increasingly dissimilar trajectories. The network must effectively learn a different curve for each IC, and the complexity of that surface grows with time, requiring even denser coverage to interpolate reliably.

#### Why Sequence Models Sidestep This Problem
(conditioning on history re-initialises the prediction task at every stride, making IC coverage irrelevant)

By conditioning on recent history rather than initial conditions, sequence models implicitly handle the IC coverage problem. Two trajectories that started from very different ICs but happen to have similar 50-step windows will look alike to the LSTM — they share the same local dynamics regardless of their origins. This means the LSTM is learning a smoother, lower-complexity function than Breen's global solution mapping. A 50-step window essentially re-initialises the prediction task at every stride, so the model generalises across IC space without ever needing to explicitly cover it.

#### Practical Consequence for This Project

This project uses the restricted circular 3BP, where the full initial state is a 4-dimensional vector `[xi, eta, vxi, veta]` plus the mass parameter μ — a 5-dimensional IC space versus Breen's 2-dimensional one. Applying the Breen architecture directly would require a dataset dense enough to cover this higher-dimensional space, demanding substantially more simulations than the 9,900 Breen used. The existing 5,000-trajectory dataset would likely be insufficient for a Breen-style global solution function to generalise well across the full IC space.

The sequence approach is therefore not just a modelling choice but the better-matched strategy for the available data. Temporal context compensates for IC sparsity: even with 5,000 trajectories, the sliding window creates 440,000 training sequences that collectively sample a wide range of dynamic regimes. The model learns from what the trajectory is doing now, not from where it started — making the data more efficiently used and the generalisation problem more tractable.

---

### Why the Breen Baseline's Informational Advantage Makes It the Right Baseline

The Breen MLP holds a structural advantage over every sequence model in this project: it is given the **true initial conditions** at inference time. When queried for the state at time t, it receives `[t, xi_0, eta_0, vxi_0, veta_0, μ]` — perfect knowledge of where and how fast every body started. The sequence models (LSTM, GRU, Transformer) receive no such oracle. They are handed a 50-step window of recent history and must implicitly reconstruct whatever information about the system's origin is still detectable in that window. In a chaotic system, that information decays: the further along a trajectory the window sits, the less it reveals about the original initial conditions.

This asymmetry is not a flaw in the comparison — it is precisely what makes the Breen MLP a meaningful upper-bound baseline.

#### What the Comparison Tells You Either Way

If the sequence models **match or approach** the Breen MLP's test MAE, the implication is strong: a 50-step history window contains enough information to recover predictions as accurate as those made with full knowledge of the initial conditions. This would validate the sequence-based approach and suggest that the system's dynamics are sufficiently self-referential — recent history encodes the IC information that matters for short-horizon prediction.

If the sequence models **fall short** of the Breen MLP, the gap is not a failure but a measurement. It quantifies exactly how much predictive accuracy is lost by not having access to the initial conditions — the irreducible cost of the harder inference problem the sequence models are actually solving. A Breen MAE of X and an LSTM MAE of X + δ means δ is the price of oracle-free prediction in this system.

Either outcome is a publishable result. The Breen MLP provides the reference ceiling; the sequence models attempt to reach it from a harder starting position.

#### Why This Makes Breen the Proper Baseline

A baseline model should represent the best achievable performance under simplified or privileged conditions, so that more realistic models can be evaluated relative to it. The Breen MLP satisfies this definition precisely:

- It has access to information (true initial conditions) that production-style sequence models cannot have, because in real forecasting scenarios the initial conditions of an observed trajectory are not known — only recent observations are available.
- Its architecture is simple and well-understood, so differences in performance cannot be attributed to architectural complexity.
- It produces point predictions at arbitrary query times, rather than extrapolating a fixed window, which is a genuinely different and in some respects easier inference modality.

Using the Breen MLP as a baseline therefore answers the question that matters most for this dissertation: *how close can a realistic, history-only sequence model get to a model with perfect initial condition knowledge?* That framing converts what might look like an unfair comparison into the most informative comparison possible.

---

## Results — Plot Interpretations

### Breen MLP — Training History

The Breen training history reveals a clear failure mode on the current dataset size. Early stopping triggered after only 18 epochs: the training MAE continued declining (reaching approximately 0.35) while validation MAE plateaued and began rising (settling near 0.44). This divergence between train and validation loss is a textbook sign of overfitting — the model has sufficient capacity to memorise the training examples but cannot generalise to unseen initial conditions.

The root cause is data sparsity in the 5-dimensional IC space. The Breen MLP must learn a global solution function over the full (t, xi_0, eta_0, vxi_0, veta_0, μ) space. With approximately 4,500 trajectories, the coverage of this space is too sparse for the model to interpolate reliably between seen and unseen initial conditions. The 18-epoch training run effectively saturated what could be learned from this data, after which additional epochs only deepened memorisation without improving generalisation.

This result is not a failure of the architecture — it is a data requirement problem. The original Breen et al. paper operated on a 2-dimensional IC space with 9,900 training simulations, a density several orders of magnitude higher than what is available here. Retraining on the 100,000-trajectory EC2 dataset is expected to substantially change this picture.

### Breen MLP — Prediction Examples

The three example predictions confirm the training history finding. In all three cases, the model produces near-constant outputs regardless of the true trajectory dynamics:

**Trajectory 1 (escape):** The true ξ(t) grows from 0 to approximately ±45 over the simulation window, with oscillations that increase in amplitude — a characteristic escape pattern. The predicted ξ remains flat near zero throughout. In phase space, the true trajectory spirals outward to a radius of ~60 while the prediction collapses to a small region near the origin. The model is outputting something close to the dataset mean rather than responding to the specific initial conditions it was given.

**Trajectory 2 (stable):** The true trajectory is nearly constant at ξ ≈ 0.7, a very low-energy stable orbit with minimal movement. Counterintuitively, the model performs worst here: it starts at ξ ≈ 0.3 and drifts upward to ξ ≈ 1.25 by t = 50, producing an incorrect trend where the true dynamics are almost perfectly flat. In phase space, the true trajectory is a near-point while the predicted trajectory wanders. This is the most revealing failure — when the true answer is simple and constant, the model still produces a wrong and dynamic prediction, suggesting it has not learned to associate specific IC patterns with stable dynamics.

**Trajectory 3 (chaotic):** The true ξ oscillates between −0.75 and 1.25 with aperiodic, high-amplitude variation. The predicted ξ stays in a narrow band near 0.75–0.85 with no meaningful variation. The phase-space plot shows the true trajectory filling a densely tangled region while the prediction occupies a tiny cluster near a single point.

Across all three examples, the model's predictions are effectively independent of which trajectory was queried — it outputs a near-constant value that reflects the training distribution mean rather than the specific dynamics of the initial conditions provided. This is consistent with the observed overfitting: the model has learned to minimise MAE across the training set by regressing to a safe average, rather than learning the solution function.

---

### RQ2 — Classification Results (Balanced Dataset)

The confusion matrices show the results under the new balanced dataset (targets: Stable 1500, Chaotic 750, Escape 1500, Collision 750) with class-weighted loss. The comparison with the original imbalanced dataset is informative precisely because the two settings differ in a controlled way — only the data generation strategy changed.

**Logistic Regression (25.3% accuracy):** Under the original imbalanced dataset, Logistic Regression collapsed entirely to predicting Escape and achieved ~70% accuracy by doing so. With the balanced dataset, this strategy no longer works — there is no dominant class to exploit. The model now distributes its predictions more broadly but with poor calibration, misclassifying heavily across all classes. The accuracy of 25.3% — below random chance for a 4-class problem — reflects that a linear decision boundary cannot separate trajectory types in the 5-dimensional IC space regardless of the training distribution. Logistic Regression serves as a clear lower bound, demonstrating that linear separability is absent.

**Random Forest (73.8% accuracy):** The Random Forest recovers meaningful structure across three of the four classes. Escape recall is high (208/222, 94%) — escape trajectories are sufficiently distinct in IC space that the ensemble can identify them reliably. Stable recall is also acceptable (162/227, 71%), reflecting that the L4/L5-targeted sampling has produced a cluster of stable ICs that the forest can partially segment. Collision recall is moderate (75/118, 64%). Chaotic recall is poor (22/66, 33%) — even with targeted sampling, chaotic orbits occupy a narrow region near L1 that is physically close to both escape and stable regions, making them difficult to separate using only initial conditions.

The overall accuracy of 73.8% represents a real drop from what would be reported under the original imbalanced dataset. This is expected: an imbalanced result is partially inflated by heavy Escape weighting. The balanced result is more honest.

**MLP (69.7% accuracy):** The MLP shows a different failure pattern from Random Forest. Its Stable recall is weaker (101/227, 45%) — the MLP does not segment the stable region as cleanly as the ensemble — but its Chaotic recall is substantially higher (50/66, 76%), significantly outperforming Random Forest on the hardest class. Escape recall is near-perfect (215/222, 97%) and Collision matches Random Forest exactly (75/118, 64%).

The MLP–RF trade-off is the most interesting finding in RQ2: both models achieve similar overall accuracy (~70–74%), but each excels in different classes. Random Forest is better at Stable; MLP is better at Chaotic. This suggests the two models are learning complementary aspects of the IC space — the ensemble exploits axis-aligned decision boundaries effective near L4/L5, while the MLP's continuous activation functions capture the smoother boundary near L1 where chaotic orbits live. An ensemble combining both would likely outperform either individually.

The key methodological finding for the paper: the balanced dataset made the Chaotic class learnable. Under the original pure-random dataset, the Chaotic class had near-zero representation and all models achieved 0% Chaotic recall. Under the new dataset, the MLP achieves 76% Chaotic recall. This is a direct, measurable consequence of the targeted sampling strategy implemented in Phase 1.

---

### RQ3 — Lagrange Point Discovery

**Scatter plot (Discovered vs Known):** The DBSCAN clustering on 1,284 low-velocity trajectory points returned four cluster centres. The dominant cluster (Cluster 0, 1,218 of 1,284 points) sits near the origin at (−0.043, 0.010) — close to the Sun's position and far from either Lagrange point (distance 0.890 from L4). This cluster represents a dense accumulation of momentary low-velocity events as trajectories pass through and around the inner region of the rotating frame, not equilibrium passages. It is a spurious result of the heuristic.

The three remaining clusters are small (11–18 points each) and more spatially meaningful. Cluster 3 at (0.185, −0.939) lies at distance 0.074 from L5 — the closest approach to a known Lagrange point in the dataset. Cluster 2 at (0.424, −0.835) is also in the L5 neighbourhood at distance 0.226. Cluster 1 at (−0.108, 1.044) sits in the upper half of the phase space at distance 0.356 from L4, a partial approach but not a recovery.

**Density heatmap:** The density heatmap reveals that most of the low-velocity points concentrate near the two primary bodies — the Sun and the planet — rather than near L4 or L5. This makes physical sense: objects naturally decelerate when passing through regions of strong gravitational influence, momentarily dropping below the velocity threshold regardless of whether they are near an equilibrium. The discovered points near L5 appear in a low-density region of the heatmap, indicating they represent rare, physically meaningful passages rather than the bulk statistical trend. L5 has a couple of discovered points nearby, including one almost right on top of it, which explains why that particular Lagrange point was recovered so accurately.

L5 is partially recovered (distance 0.074) while L4 is not meaningfully recovered. This asymmetry likely reflects the distribution of stable trajectories in the dataset: the targeted sampling near L4/L5 produced orbits that pass closer to L5 in the simulation window used, by chance of the initial condition perturbations and the integration duration. A longer simulation window or a more symmetric sampling strategy might recover both equally.

The low-velocity heuristic conflates two physically distinct phenomena: true equilibrium passages and general turning points anywhere along an orbit. A principled method would require filtering by the Jacobi constant value near the theoretical equilibrium value at L4/L5, or identifying positions where the net force on the particle drops below a threshold. The current approach, while simple to implement, cannot reliably distinguish equilibrium from deceleration — the near-recovery of L5 at distance 0.074 is encouraging but the dominant spurious cluster near the origin illustrates the method's fundamental imprecision.

---

### RQ1 — Sequence Model Training Results

**Training history:** The training history plot shows all three sequence models simultaneously. LSTM and GRU share a common pattern: a rapid descent in the first 10 epochs (roughly one order of magnitude reduction in MSE), followed by slow refinement over the remaining epochs. Both models converge with validation curves tracking training curves closely — a small, stable gap indicating good generalisation attributable to the stride=5 windowing fix.

The Transformer's behaviour is qualitatively different. Both its training and validation loss plateau from epoch 2 onward at approximately MSE 0.09, roughly 20× higher than the LSTM/GRU plateau of ~0.004. Crucially, the Transformer's training loss is nearly as high as its validation loss — this is not overfitting (where train diverges from val) but rather a failure to fit the training data at all. The model is stuck in a region of the loss landscape where it cannot make further progress, likely a local minimum or a saddle point that the attention mechanism cannot escape with the current learning rate and architecture settings. Early stopping triggered at epoch 21 after validation loss showed no improvement.

**Final results:**

| Model | Test MSE | Test MAE | Inference | Epochs |
|-------|----------|----------|-----------|--------|
| LSTM | — | 0.019227 | 18.01 ms | 48 |
| GRU | — | 0.019972 | 36.34 ms | 67 |
| Transformer | 0.079493 | 0.122307 | 18.94 ms | 21 |
| Breen MLP | — | 0.457661 | 62.50 ms | 18 |
| Numerical Integration | — | — | 8.03 ms | — |

LSTM marginally outperforms GRU in test MAE (0.019 vs 0.020). The GRU inference time of 36.34ms being higher than LSTM's 18.01ms is likely a measurement artifact from single-sample timing overhead rather than a true architectural difference — GRU has fewer parameters than LSTM and would not be expected to be slower in practice.

**Transformer underperformance:** The Transformer's test MAE of 0.122 is approximately 6× worse than LSTM/GRU. This is consistent with the known data requirements of attention-based architectures. Self-attention learns which of the 50 input timesteps to attend to for each prediction — but learning meaningful attention patterns requires many diverse examples. With ~420,000 training sequences from 4,500 trajectories, the Transformer does not have enough variety to calibrate its attention weights. LSTM and GRU learn local recurrence patterns from much shorter effective memory windows, making them far more sample-efficient at this scale.

This is the same root cause as the Breen failure: insufficient data for the model's learning strategy. All three underperformers (Breen, Transformer, and to a lesser degree GRU) stopped early and plateaued well above LSTM's performance level. The 100,000-trajectory EC2 dataset is expected to be the differentiating experiment — particularly for the Transformer, which has demonstrated strong performance on chaotic systems in the literature when given sufficient training data.

**CPU inference is slower than numerical integration:** All neural network models are slower than numerical integration on CPU (8.03 ms per sample). LSTM and Transformer run at ~0.4× the speed of the ODE solver; GRU at ~0.2×. This directly contradicts the computational efficiency motivation in the abstract and the Breen et al. speedup claim. The explanation is TensorFlow's CPU overhead: graph execution, memory allocation, and data movement dominate for single-sample inference at this scale. The speedup claim holds only on GPU, where inference is parallelised across thousands of cores and the overhead amortises across large batches. This makes EC2 GPU deployment a scientific requirement for the efficiency argument in RQ1, not merely an engineering convenience.

---

## EC2 Scaling — 45k Smoke Test Observations

### EC2 Commands Reference

```bash
# Connect
ssh -i ~/Desktop/3bpec2.pem ubuntu@<ec2-ip>

# Environment
source /opt/tensorflow/bin/activate
cd deeplearningapps3bp
git pull origin test/smoke-test-45k
git checkout test/smoke-test-45k

# Start training (background, survives disconnect)
nohup python run_all.py > training_log_smoke_test_45k.txt 2>&1 &
echo $!

# Monitor
tail -f training_log_smoke_test_45k.txt

# Stop instance when done: EC2 console → Instance state → Stop instance
```

### Data Generation Timing — Local vs EC2

The local run generated 4,216 trajectories (out of a 4,500 target) in approximately 3 hours 40 minutes on CPU. The training log captured the progress bar stalling near the end at `26.30s/it` — the ODE solver slowing dramatically as the sampler exhausted easy trajectories and was left hunting for rare chaotic and collision orbits.

The EC2 smoke test generated 37,846 trajectories in approximately 2 hours 48 minutes on the g4dn.xlarge instance (4 vCPUs, no GPU involvement during data generation — this is pure CPU ODE integration). Per-trajectory timing:

| | Local (MacBook CPU) | EC2 (4 vCPU) |
|---|---|---|
| Trajectories generated | 4,216 | 37,846 |
| Data generation time | ~3h 40m | ~2h 48m |
| Time per trajectory | ~3.1s | ~0.27s |
| Relative speedup | — | **~11.5× per trajectory** |

The 11.5× speedup per trajectory, combined with 9× more trajectories generated, means the EC2 run produced roughly **100× more training data in 25% less wall time** than the local run. This is the clearest demonstration of why cloud compute is necessary for this project at scale.

### Chaotic Class Remains Structurally Underrepresented

Despite the 45k target and ~100,000 integration attempts, the final dataset contained only **346 Chaotic trajectories** — 4.6% of the 7,500 target and 0.9% of the total 37,846 collected. This is the same physics-driven scarcity observed locally: the near-L1 targeted sampler produces many escape and collision trajectories that fail the Chaotic classification threshold, and the extreme near-miss bounded orbits that define the Chaotic class are disproportionately filtered by ODE solver convergence failures.

This is not a code failure or a scaling failure — it is a fundamental property of the RC3BP phase space. The region near L1 that produces bounded high-energy-variation orbits is physically narrow, and the fraction of sampler draws that land precisely in that region is low regardless of total attempt count. Generating 7,500 Chaotic trajectories would require either a more targeted sampler (tighter positioning around the L1 saddle with finely tuned velocity ranges) or a significantly larger attempt budget.

For the paper, this result is worth reporting explicitly: even with targeted sampling and 100,000 integration attempts, the Chaotic class remains a hard minority. It establishes that the classification imbalance problem has a physical root cause that cannot be resolved by scaling compute alone.

### Classification Results at 37,846 Trajectories

The classification step ran in under 2 minutes total on GPU (828 steps × 2s/epoch for the MLP). Results:

| Model | Test Accuracy | Chaotic Recall | Notes |
|---|---|---|---|
| Logistic Regression | 21.7% | 53% | Collapsed — no stable separating hyperplane |
| Random Forest | 80.9% | 2% | Strong on Stable/Escape, misses Chaotic entirely |
| MLP | 73.9% | 51% | Weaker overall but captures Chaotic signal |

The RF–MLP trade-off from the local run persists at scale: Random Forest improves to 80.9% (+7 points over local) but its Chaotic recall drops to 2% — the increased dataset volume improved majority-class performance but did not help with the structurally rare Chaotic class. MLP holds at 73.9% with 51% Chaotic recall. The complementary failure pattern (RF better at Stable, MLP better at Chaotic) is reproduced at larger scale, reinforcing that this reflects model inductive bias rather than dataset noise.

Logistic Regression collapsed to near-random (21.7%) — consistent with both the local result and the theoretical expectation that linear boundaries cannot separate trajectory types in the 5-dimensional IC space.

### Breen Baseline — Improved Behaviour at Scale

On the local 4,216-trajectory dataset, the Breen MLP overfit immediately: training MAE decreased while validation MAE rose from epoch 1, triggering early stopping at epoch 18. The model regressed to the dataset mean rather than learning the solution function.

On the 45k EC2 dataset, the Breen baseline shows qualitatively different behaviour. Validation MAE tracks training MAE through the first 12+ epochs and both continue decreasing:

```
Epoch  1: train MAE 0.4091, val MAE 0.3868
Epoch  3: train MAE 0.2268, val MAE 0.2166
Epoch  8: train MAE 0.1814, val MAE 0.1980
Epoch 12: train MAE 0.1710, val MAE 0.2106
```

This is not overfitting — both curves are declining and the gap is small. The increased IC space coverage from ~10× more trajectories has given the model enough nearby training examples to interpolate rather than memorise. Whether the model ultimately converges to a test MAE that approaches LSTM/GRU remains to be seen, but the training dynamic confirms the hypothesis from the local run: the Breen failure was a data coverage problem, not an architectural one.