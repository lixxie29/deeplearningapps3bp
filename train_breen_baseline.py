"""
Training Script for Breen et al. (2019) Baseline (RQ1 comparison)

Trains a 10-layer MLP that maps (t, IC, mu) → state_at_t directly,
without any sequence context. Results are saved alongside LSTM/GRU/Transformer
results for direct comparison in the unified test suite.
"""

import numpy as np
import pickle
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

from preprocessing import DataPreprocessor
from models import build_breen_mlp


def train_breen_baseline():
    print("Loading and preparing data...")
    preprocessor = DataPreprocessor('three_body_dataset.pkl')
    data = preprocessor.prepare_breen_data()

    X_train, y_train = data['X_train'], data['y_train']
    X_val,   y_val   = data['X_val'],   data['y_val']
    X_test,  y_test  = data['X_test'],  data['y_test']

    print("="*50)
    print("Training Breen MLP Baseline")
    print("="*50)

    model = build_breen_mlp(input_dim=X_train.shape[1])
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7, verbose=1
        ),
    ]

    start = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10000,
        batch_size=5000,
        callbacks=callbacks,
        verbose=1,
    )
    train_time = time.time() - start

    # Evaluation
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest MAE:       {test_mae:.6f}")
    print(f"Training time:  {train_time:.1f}s")

    # Inference timing — 100 single-sample passes, skip first 10 (warm-up)
    single = X_test[:1]
    _ = model.predict(single, verbose=0)  # warm-up
    times = []
    for _ in range(110):
        t0 = time.time()
        model.predict(single, verbose=0)
        times.append(time.time() - t0)
    inference_ms = np.mean(times[10:]) * 1000
    print(f"Inference time: {inference_ms:.2f} ms/sample")

    # Save model
    model.save('breen_model.keras')

    # Save results in the same format as prediction_results.pkl
    results = {
        'Breen MLP': {
            'history':        history.history,
            'test_loss':      test_loss,
            'test_mae':       test_mae,
            'inference_time': inference_ms / 1000,  # seconds, matches prediction_results
            'train_time_s':   train_time,
        }
    }
    with open('breen_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("Results saved to breen_results.pkl")

    # Training history plot
    plt.figure(figsize=(9, 4))
    epochs_ran = range(1, len(history.history['loss']) + 1)
    plt.plot(epochs_ran, history.history['loss'],     label='Train MAE')
    plt.plot(epochs_ran, history.history['val_loss'], label='Val MAE', linestyle='--')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('MAE (log scale)')
    plt.title('Breen MLP — Training History')
    plt.legend()
    plt.tight_layout()
    plt.savefig('breen_training_history.png', dpi=150)
    plt.close()
    print("Saved breen_training_history.png")

    # Example predictions: pick 3 test trajectories, show t-indexed predictions
    _plot_examples(model, data)

    return results


def _plot_examples(model, data):
    """
    For 3 random test trajectories, plot predicted vs true state at every
    timestep — treating each (t, IC) query independently as the Breen model does.
    """
    rng = np.random.default_rng(42)
    scaler_X = data['scaler_X']
    scaler_y = data['scaler_y']

    # Recover original (un-normalised) arrays to find trajectory boundaries
    X_raw = scaler_X.inverse_transform(data['X_test'])   # (N, 6)
    y_raw = scaler_y.inverse_transform(data['y_test'])   # (N, 4)

    # Group samples back into trajectories by their initial condition (IC fingerprint)
    # ic_key = (xi_0, eta_0, vxi_0, veta_0, mu) — unique per trajectory
    ic_keys = [tuple(np.round(X_raw[i, 1:], 6)) for i in range(len(X_raw))]
    from collections import defaultdict
    traj_map = defaultdict(list)
    for i, key in enumerate(ic_keys):
        traj_map[key].append(i)

    traj_keys = list(traj_map.keys())
    chosen = rng.choice(len(traj_keys), size=min(3, len(traj_keys)), replace=False)

    fig, axes = plt.subplots(2, len(chosen), figsize=(5 * len(chosen), 8))
    if len(chosen) == 1:
        axes = axes[:, np.newaxis]

    for col, idx in enumerate(chosen):
        key   = traj_keys[idx]
        idxs  = sorted(traj_map[key], key=lambda i: X_raw[i, 0])  # sort by t
        t_vals = X_raw[idxs, 0]

        X_norm = data['X_test'][idxs]
        y_pred_norm = model.predict(X_norm, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_norm)
        y_true = y_raw[idxs]

        # Row 0: xi trajectory over time
        axes[0, col].plot(t_vals, y_true[:, 0], 'g-',  label='True ξ',      linewidth=1.5)
        axes[0, col].plot(t_vals, y_pred[:, 0], 'b--', label='Predicted ξ', linewidth=1.5)
        axes[0, col].set_xlabel('t')
        axes[0, col].set_ylabel('ξ')
        axes[0, col].set_title(f'Trajectory {col+1} — ξ(t)')
        axes[0, col].legend()
        axes[0, col].grid(True, alpha=0.3)

        # Row 1: phase-space orbit (xi vs eta)
        axes[1, col].plot(y_true[:, 0], y_true[:, 1], 'g-',  label='True',      linewidth=1.5)
        axes[1, col].plot(y_pred[:, 0], y_pred[:, 1], 'b--', label='Predicted', linewidth=1.5)
        axes[1, col].set_xlabel('ξ')
        axes[1, col].set_ylabel('η')
        axes[1, col].set_title(f'Trajectory {col+1} — Phase space')
        axes[1, col].legend()
        axes[1, col].grid(True, alpha=0.3)
        axes[1, col].axis('equal')

    plt.tight_layout()
    plt.savefig('breen_prediction_examples.png', dpi=150)
    plt.close()
    print("Saved breen_prediction_examples.png")


if __name__ == "__main__":
    train_breen_baseline()
