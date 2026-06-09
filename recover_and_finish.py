"""
Recovery script: train only iTransformer, then load all 5 saved models,
evaluate them, generate plots, and save prediction_results.pkl.
Run this instead of re-running train_prediction.py from scratch.
"""

import numpy as np
import pickle
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from preprocessing import DataPreprocessor
from models import build_itransformer_predictor

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

print("Loading and preprocessing data...")
preprocessor = DataPreprocessor('three_body_dataset.pkl')
data = preprocessor.prepare_prediction_data(input_length=50, output_length=10)

# ── 1. Train iTransformer ──────────────────────────────────────────────────
print("\n" + "="*50)
print("Training iTransformer Predictor")
print("="*50)

itransformer = build_itransformer_predictor(input_length=50, output_length=10)
history_itransformer = itransformer.fit(
    data['X_train'], data['y_train'],
    validation_data=(data['X_val'], data['y_val']),
    epochs=100,
    batch_size=32,
    verbose=1,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
        tf.keras.callbacks.CSVLogger('itransformer_training_log.csv'),
    ],
)
itransformer.save('itransformer_model.keras')
print("iTransformer saved.")

# ── 2. Load pre-trained models ─────────────────────────────────────────────
print("\nLoading saved models...")
saved_models = {
    'LSTM':                  tf.keras.models.load_model('lstm_model.keras'),
    'GRU':                   tf.keras.models.load_model('gru_model.keras'),
    'Transformer':           tf.keras.models.load_model('transformer_model.keras'),
    'Transformer (revised)': tf.keras.models.load_model('transformer_revised_model.keras'),
}

saved_histories = {}
for name, csv_name in [
    ('LSTM',                  'lstm_training_log.csv'),
    ('GRU',                   'gru_training_log.csv'),
    ('Transformer',           'transformer_training_log.csv'),
    ('Transformer (revised)', 'transformer_revised_training_log.csv'),
]:
    import csv
    h = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}
    with open(csv_name) as f:
        for row in csv.DictReader(f):
            h['loss'].append(float(row['loss']))
            h['val_loss'].append(float(row['val_loss']))
            h['mae'].append(float(row['mae']))
            h['val_mae'].append(float(row['val_mae']))
    saved_histories[name] = h

# ── 3. Evaluate all models ─────────────────────────────────────────────────
results = {}

all_models = {**saved_models, 'iTransformer': itransformer}
all_histories = {**saved_histories, 'iTransformer': history_itransformer.history}

for name, model in all_models.items():
    test_loss, test_mae = model.evaluate(data['X_test'], data['y_test'], verbose=0)
    print(f"{name}: MSE={test_loss:.6f}  MAE={test_mae:.6f}")

    start = time.time()
    preds = model.predict(data['X_test'][:100], verbose=0)
    inf_time = (time.time() - start) / 100

    results[name] = {
        'history':        all_histories[name],
        'test_loss':      test_loss,
        'test_mae':       test_mae,
        'predictions':    preds,
        'inference_time': inf_time,
    }

# ── 4. Numerical integration baseline time ────────────────────────────────
from data_generation import ThreeBodyDataGenerator
generator = ThreeBodyDataGenerator()
start = time.time()
for i in range(10):
    initial_state_scaled = data['X_test'][i, -1, :]
    initial_state = data['scaler'].inverse_transform(initial_state_scaled.reshape(1, -1))[0]
    generator.generate_single_trajectory(initial_state, mu=0.3, t_max=1, n_points=10)
numerical_time = (time.time() - start) / 10

print("\nInference Time Comparison:")
for name, r in results.items():
    speedup = numerical_time / r['inference_time']
    print(f"  {name:<24}: {r['inference_time']*1000:.3f} ms  ({speedup:.1f}x vs numerical)")
print(f"  {'Numerical':<24}: {numerical_time*1000:.3f} ms")

# ── 5. Training history plot ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for name, r in results.items():
    axes[0].plot(r['history']['loss'],     label=f'{name} Train')
    axes[0].plot(r['history']['val_loss'], label=f'{name} Val', linestyle='--')
axes[0].set(xlabel='Epoch', ylabel='Loss (MSE)', title='Training History - Loss')
axes[0].legend(fontsize=7)
axes[0].set_yscale('log')
axes[0].grid(True)

for name, r in results.items():
    axes[1].plot(r['history']['mae'],     label=f'{name} Train')
    axes[1].plot(r['history']['val_mae'], label=f'{name} Val', linestyle='--')
axes[1].set(xlabel='Epoch', ylabel='MAE', title='Training History - MAE')
axes[1].legend(fontsize=7)
axes[1].grid(True)

plt.tight_layout()
plt.savefig('prediction_training_history.png', dpi=300, bbox_inches='tight')
print("\nSaved prediction_training_history.png")
plt.close()

# ── 6. Example predictions plot ───────────────────────────────────────────
model_styles = {
    'LSTM':                  ('b', '--', 's'),
    'GRU':                   ('r', ':',  '^'),
    'Transformer':           ('m', '-.', 'D'),
    'Transformer (revised)': ('g', '--', 'P'),
    'iTransformer':          ('c', '--', 'o'),
}
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for idx in range(3):
    X_sample  = data['X_test'][idx].copy()
    y_true    = data['y_test'][idx].copy()
    scaler    = data['scaler']
    X_plot    = scaler.inverse_transform(X_sample)
    y_true_plot = scaler.inverse_transform(y_true)

    axes[0, idx].plot(X_plot[:, 0],      X_plot[:, 1],      'k-', label='Input', linewidth=2)
    axes[0, idx].plot(y_true_plot[:, 0], y_true_plot[:, 1], 'g-', label='True',  linewidth=2, marker='o')

    time_input  = np.arange(50)
    time_output = np.arange(50, 60)
    axes[1, idx].plot(time_input,  X_plot[:, 0],      'k-', label='Input', linewidth=2)
    axes[1, idx].plot(time_output, y_true_plot[:, 0], 'g-', label='True',  linewidth=2, marker='o')

    for name, (color, ls, marker) in model_styles.items():
        y_pred = all_models[name].predict(X_sample[np.newaxis, ...], verbose=0)[0]
        y_pred_plot = scaler.inverse_transform(y_pred)
        axes[0, idx].plot(y_pred_plot[:, 0], y_pred_plot[:, 1],
                          color=color, linestyle=ls, marker=marker, label=name, linewidth=1.5)
        axes[1, idx].plot(time_output, y_pred_plot[:, 0],
                          color=color, linestyle=ls, marker=marker, label=name, linewidth=1.5)

    axes[0, idx].set(xlabel='ξ', ylabel='η', title=f'Sample {idx+1} - Position')
    axes[0, idx].legend(fontsize=7)
    axes[0, idx].grid(True, alpha=0.3)
    axes[0, idx].axis('equal')

    axes[1, idx].axvline(x=50, color='gray', linestyle='--', alpha=0.5)
    axes[1, idx].set(xlabel='Time Step', ylabel='ξ', title=f'Sample {idx+1} - ξ Coordinate')
    axes[1, idx].legend(fontsize=7)
    axes[1, idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('prediction_examples.png', dpi=300, bbox_inches='tight')
print("Saved prediction_examples.png")
plt.close()

# ── 7. Save results pickle ─────────────────────────────────────────────────
save_results = {
    name: {
        'history':        r['history'],
        'test_loss':      r['test_loss'],
        'test_mae':       r['test_mae'],
        'inference_time': r['inference_time'],
    }
    for name, r in results.items()
}
with open('prediction_results.pkl', 'wb') as f:
    pickle.dump(save_results, f)
print("Saved prediction_results.pkl")

# ── 8. Upload everything to S3 ────────────────────────────────────────────
from s3_utils import upload_all_results
upload_all_results()
print("\nDone.")
