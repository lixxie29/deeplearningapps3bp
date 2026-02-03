# Three-Body Problem Deep Learning Project

Deep learning approaches for solving the three-body problem: trajectory prediction, stability classification, and equilibrium discovery.

## Research Questions

1. **RQ1**: Can LSTM/GRU networks learn to predict three-body system trajectories more efficiently than traditional numerical integration methods?
2. **RQ2**: Can deep learning classify the stability of three-body configurations from initial conditions?
3. **RQ3**: Can neural networks discover Lagrangian equilibrium points without explicit physics equations?

## Setup

### Requirements
```bash
pip install -r requirements.txt
```

### Quick Start
```bash
# Run all experiments
python run_all.py
```

This will:
1. Generate 5000 trajectories (~30-60 minutes)
2. Train classification models (~10 minutes)
3. Train prediction models (~30-60 minutes)
4. Discover equilibrium points (~5 minutes)

### Run Individual Components
```bash
# Generate data only
python data_generation.py

# Train classification models (RQ2)
python train_classification.py

# Train prediction models (RQ1)
python train_prediction.py

# Discover Lagrange points (RQ3)
python discover_equilibria.py
```

## Project Structure
```
.
├── requirements.txt              # Dependencies
├── data_generation.py           # Generate trajectory dataset
├── preprocessing.py             # Prepare data for ML
├── models.py                    # ML/DL model definitions
├── train_classification.py      # RQ2: Stability classification
├── train_prediction.py          # RQ1: Trajectory prediction
├── discover_equilibria.py       # RQ3: Equilibrium discovery
├── run_all.py                   # Main workflow script
└── README.md                    # This file
```

## Outputs

### Data Files
- `three_body_dataset.pkl` - Generated trajectories
- `classification_results.pkl` - Classification model results
- `prediction_results.pkl` - Prediction model results
- `equilibrium_discovery_results.pkl` - Discovered equilibrium points

### Visualizations
- `classification_confusion_matrices.png` - Confusion matrices for all classifiers
- `prediction_training_history.png` - Training curves for LSTM/GRU
- `prediction_examples.png` - Example trajectory predictions
- `lagrange_point_discovery.png` - Discovered vs known Lagrange points

## Expected Results

### RQ2 - Classification
- Random Forest: ~80-85% accuracy
- MLP: ~82-87% accuracy
- Best at distinguishing stable vs chaotic/escape trajectories

### RQ1 - Prediction
- LSTM/GRU: Good predictions for 10-20 timesteps
- 10-100x faster than numerical integration
- Accuracy degrades for long-term chaotic behavior

### RQ3 - Equilibrium Discovery
- Should discover 2 clusters near L4, L5 Lagrange points
- Distance to known points: <0.2 units
- Demonstrates unsupervised learning of physics

## Theoretical Background

This project is based on the restricted circular three-body problem from celestial mechanics:

- **Equations of motion** (5.10): Synodic frame differential equations
- **Jacobi integral** (5.12): Conserved quantity in rotating frame
- **Lagrange points** (Section 3.4): Equilibrium solutions

## Citation

If you use this code, please cite:
```
Ciocan, C.M. (2024). The 2 and 3 Body Problems. 
Bachelor's Thesis, Babeș-Bolyai University.
```

## License

MIT License