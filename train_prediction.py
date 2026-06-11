"""
Training Script for Prediction Task (RQ1)
Trains LSTM/GRU/Transformer/iTransformer models to predict future trajectory points
"""

import numpy as np
import pickle
import time
from preprocessing import DataPreprocessor
from models import build_lstm_predictor, build_gru_predictor, build_transformer_predictor, build_transformer_predictor_revised, build_itransformer_predictor
import matplotlib.pyplot as plt
import tensorflow as tf

def _prediction_callbacks(name):
    return [
        tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
        tf.keras.callbacks.CSVLogger(f'{name.lower()}_training_log.csv'),
    ]

def train_prediction_models(smoke=False, seed=42):
    """Train trajectory prediction models"""
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Load data
    print("Loading and preprocessing data...")
    preprocessor = DataPreprocessor('three_body_dataset.pkl')
    data = preprocessor.prepare_prediction_data(input_length=50, output_length=10)

    if smoke:
        data['X_train'] = data['X_train'][:500]
        data['y_train'] = data['y_train'][:500]
    _epochs = 2 if smoke else 100

    results = {}

    # 1. LSTM
    print("\n" + "="*50)
    print("Training LSTM Predictor")
    print("="*50)

    lstm = build_lstm_predictor(input_length=50, output_length=10)

    history_lstm = lstm.fit(
        data['X_train'], data['y_train'],
        validation_data=(data['X_val'], data['y_val']),
        epochs=_epochs,
        batch_size=32,
        verbose=1,
        callbacks=_prediction_callbacks('lstm'),
    )
    lstm.save('lstm_model.keras')

    test_loss, test_mae = lstm.evaluate(data['X_test'], data['y_test'], verbose=0)
    print(f"\nLSTM Test Loss (MSE): {test_loss:.6f}")
    print(f"LSTM Test MAE: {test_mae:.6f}")

    start = time.time()
    predictions_lstm = lstm.predict(data['X_test'][:100], verbose=0)
    lstm_time = time.time() - start

    results['LSTM'] = {
        'model': lstm,
        'history': history_lstm.history,
        'test_loss': test_loss,
        'test_mae': test_mae,
        'predictions': predictions_lstm,
        'inference_time': lstm_time / 100,
    }

    # 2. GRU
    print("\n" + "="*50)
    print("Training GRU Predictor")
    print("="*50)

    gru = build_gru_predictor(input_length=50, output_length=10)

    history_gru = gru.fit(
        data['X_train'], data['y_train'],
        validation_data=(data['X_val'], data['y_val']),
        epochs=_epochs,
        batch_size=32,
        verbose=1,
        callbacks=_prediction_callbacks('gru'),
    )
    gru.save('gru_model.keras')

    test_loss, test_mae = gru.evaluate(data['X_test'], data['y_test'], verbose=0)
    print(f"\nGRU Test Loss (MSE): {test_loss:.6f}")
    print(f"GRU Test MAE: {test_mae:.6f}")

    start = time.time()
    predictions_gru = gru.predict(data['X_test'][:100], verbose=0)
    gru_time = time.time() - start

    results['GRU'] = {
        'model': gru,
        'history': history_gru.history,
        'test_loss': test_loss,
        'test_mae': test_mae,
        'predictions': predictions_gru,
        'inference_time': gru_time / 100,
    }

    # 3. Transformer
    print("\n" + "="*50)
    print("Training Transformer Predictor")
    print("="*50)

    transformer = build_transformer_predictor(input_length=50, output_length=10)

    history_transformer = transformer.fit(
        data['X_train'], data['y_train'],
        validation_data=(data['X_val'], data['y_val']),
        epochs=_epochs,
        batch_size=32,
        verbose=1,
        callbacks=_prediction_callbacks('transformer'),
    )
    transformer.save('transformer_model.keras')

    test_loss, test_mae = transformer.evaluate(data['X_test'], data['y_test'], verbose=0)
    print(f"\nTransformer Test Loss (MSE): {test_loss:.6f}")
    print(f"Transformer Test MAE: {test_mae:.6f}")

    start = time.time()
    predictions_transformer = transformer.predict(data['X_test'][:100], verbose=0)
    transformer_time = time.time() - start

    results['Transformer'] = {
        'model': transformer,
        'history': history_transformer.history,
        'test_loss': test_loss,
        'test_mae': test_mae,
        'predictions': predictions_transformer,
        'inference_time': transformer_time / 100,
    }

    # 4. Transformer (revised) — ablation for Section 6.2.4
    print("\n" + "="*50)
    print("Training Transformer (revised) Predictor")
    print("="*50)

    transformer_rev = build_transformer_predictor_revised(input_length=50, output_length=10)

    history_transformer_rev = transformer_rev.fit(
        data['X_train'], data['y_train'],
        validation_data=(data['X_val'], data['y_val']),
        epochs=_epochs,
        batch_size=32,
        verbose=1,
        callbacks=_prediction_callbacks('transformer_revised'),
    )
    transformer_rev.save('transformer_revised_model.keras')

    test_loss, test_mae = transformer_rev.evaluate(data['X_test'], data['y_test'], verbose=0)
    print(f"\nTransformer (revised) Test Loss (MSE): {test_loss:.6f}")
    print(f"Transformer (revised) Test MAE: {test_mae:.6f}")

    start = time.time()
    predictions_transformer_rev = transformer_rev.predict(data['X_test'][:100], verbose=0)
    transformer_rev_time = time.time() - start

    results['Transformer (revised)'] = {
        'model': transformer_rev,
        'history': history_transformer_rev.history,
        'test_loss': test_loss,
        'test_mae': test_mae,
        'predictions': predictions_transformer_rev,
        'inference_time': transformer_rev_time / 100,
    }

    # 5. iTransformer
    print("\n" + "="*50)
    print("Training iTransformer Predictor")
    print("="*50)

    itransformer = build_itransformer_predictor(input_length=50, output_length=10)

    history_itransformer = itransformer.fit(
        data['X_train'], data['y_train'],
        validation_data=(data['X_val'], data['y_val']),
        epochs=_epochs,
        batch_size=32,
        verbose=1,
        callbacks=_prediction_callbacks('itransformer'),
    )
    itransformer.save('itransformer_model.keras')

    test_loss, test_mae = itransformer.evaluate(data['X_test'], data['y_test'], verbose=0)
    print(f"\niTransformer Test Loss (MSE): {test_loss:.6f}")
    print(f"iTransformer Test MAE: {test_mae:.6f}")

    start = time.time()
    predictions_itransformer = itransformer.predict(data['X_test'][:100], verbose=0)
    itransformer_time = time.time() - start

    results['iTransformer'] = {
        'model': itransformer,
        'history': history_itransformer.history,
        'test_loss': test_loss,
        'test_mae': test_mae,
        'predictions': predictions_itransformer,
        'inference_time': itransformer_time / 100,
    }

    # 5. Compare with numerical integration time
    print("\n" + "="*50)
    print("Comparing with Numerical Integration")
    print("="*50)

    from data_generation import ThreeBodyDataGenerator
    generator = ThreeBodyDataGenerator()

    start = time.time()
    for i in range(10):
        initial_state_scaled = data['X_test'][i, -1, :]
        initial_state = data['scaler'].inverse_transform(initial_state_scaled.reshape(1, -1))[0]
        generator.generate_single_trajectory(initial_state, mu=0.3, t_max=1, n_points=10)
    numerical_time = (time.time() - start) / 10

    print(f"\nInference Time Comparison (per sample):")
    for name, result in results.items():
        speedup = numerical_time / result['inference_time']
        print(f"  {name:<14}: {result['inference_time']*1000:.3f} ms  ({speedup:.1f}x vs numerical)")
    print(f"  {'Numerical':<14}: {numerical_time*1000:.3f} ms")

    # 6. Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for name, result in results.items():
        axes[0].plot(result['history']['loss'], label=f'{name} Train')
        axes[0].plot(result['history']['val_loss'], label=f'{name} Val', linestyle='--')

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Training History - Loss')
    axes[0].legend()
    axes[0].set_yscale('log')
    axes[0].grid(True)

    for name, result in results.items():
        axes[1].plot(result['history']['mae'], label=f'{name} Train')
        axes[1].plot(result['history']['val_mae'], label=f'{name} Val', linestyle='--')

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Training History - MAE')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('prediction_training_history.png', dpi=300, bbox_inches='tight')
    print("\nTraining history saved to 'prediction_training_history.png'")
    plt.close()

    # 7. Visualize example predictions
    model_styles = {
        'LSTM':                 ('b', '--', 's'),
        'GRU':                  ('r', ':',  '^'),
        'Transformer':          ('m', '-.', 'D'),
        'Transformer (revised)':('g', '--', 'P'),
        'iTransformer':         ('c', '--', 'o'),
    }
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for idx in range(3):
        X_sample = data['X_test'][idx].copy()
        y_true = data['y_test'][idx].copy()
        scaler = data['scaler']

        X_plot = scaler.inverse_transform(X_sample)
        y_true_plot = scaler.inverse_transform(y_true)

        axes[0, idx].plot(X_plot[:, 0], X_plot[:, 1], 'k-', label='Input', linewidth=2)
        axes[0, idx].plot(y_true_plot[:, 0], y_true_plot[:, 1], 'g-', label='True', linewidth=2, marker='o')

        time_input = np.arange(50)
        time_output = np.arange(50, 60)
        axes[1, idx].plot(time_input, X_plot[:, 0], 'k-', label='Input', linewidth=2)
        axes[1, idx].plot(time_output, y_true_plot[:, 0], 'g-', label='True', linewidth=2, marker='o')

        for name, (color, ls, marker) in model_styles.items():
            y_pred = results[name]['model'].predict(X_sample[np.newaxis, ...], verbose=0)[0]
            y_pred_plot = scaler.inverse_transform(y_pred)
            axes[0, idx].plot(y_pred_plot[:, 0], y_pred_plot[:, 1],
                              color=color, linestyle=ls, marker=marker, label=name, linewidth=1.5)
            axes[1, idx].plot(time_output, y_pred_plot[:, 0],
                              color=color, linestyle=ls, marker=marker, label=name, linewidth=1.5)

        axes[0, idx].set_xlabel('ξ')
        axes[0, idx].set_ylabel('η')
        axes[0, idx].set_title(f'Sample {idx+1} - Position')
        axes[0, idx].legend(fontsize=7)
        axes[0, idx].grid(True, alpha=0.3)
        axes[0, idx].axis('equal')

        axes[1, idx].axvline(x=50, color='gray', linestyle='--', alpha=0.5)
        axes[1, idx].set_xlabel('Time Step')
        axes[1, idx].set_ylabel('ξ')
        axes[1, idx].set_title(f'Sample {idx+1} - ξ Coordinate')
        axes[1, idx].legend(fontsize=7)
        axes[1, idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('prediction_examples.png', dpi=300, bbox_inches='tight')
    print("Example predictions saved to 'prediction_examples.png'")
    plt.close()

    # Save results
    save_results = {}
    for name, result in results.items():
        save_results[name] = {
            'history': result['history'],
            'test_loss': result['test_loss'],
            'test_mae': result['test_mae'],
            'inference_time': result['inference_time'],
        }

    with open('prediction_results.pkl', 'wb') as f:
        pickle.dump(save_results, f)

    print("\nResults saved to 'prediction_results.pkl'")

    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true', help='2-epoch smoke test')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    train_prediction_models(smoke=args.smoke, seed=args.seed)
