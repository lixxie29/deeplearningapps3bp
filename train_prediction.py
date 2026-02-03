"""
Training Script for Prediction Task (RQ1)
Trains LSTM/GRU models to predict future trajectory points
"""

import numpy as np
import pickle
import time
from preprocessing import DataPreprocessor
from models import build_lstm_predictor, build_gru_predictor
import matplotlib.pyplot as plt
import tensorflow as tf

def train_prediction_models():
    """Train trajectory prediction models"""
    
    # Load data
    print("Loading and preprocessing data...")
    preprocessor = DataPreprocessor('three_body_dataset.pkl')
    data = preprocessor.prepare_prediction_data(input_length=50, output_length=10)
    
    results = {}
    
    # 1. LSTM
    print("\n" + "="*50)
    print("Training LSTM Predictor")
    print("="*50)
    
    lstm = build_lstm_predictor(input_length=50, output_length=10)
    
    history_lstm = lstm.fit(
        data['X_train'], data['y_train'],
        validation_data=(data['X_val'], data['y_val']),
        epochs=100,
        batch_size=32,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
        ]
    )
    
    # Evaluate
    test_loss, test_mae = lstm.evaluate(data['X_test'], data['y_test'], verbose=0)
    
    print(f"\nLSTM Test Loss (MSE): {test_loss:.6f}")
    print(f"LSTM Test MAE: {test_mae:.6f}")
    
    # Timing
    start = time.time()
    predictions_lstm = lstm.predict(data['X_test'][:100], verbose=0)
    lstm_time = time.time() - start
    
    results['LSTM'] = {
        'model': lstm,
        'history': history_lstm.history,
        'test_loss': test_loss,
        'test_mae': test_mae,
        'predictions': predictions_lstm,
        'inference_time': lstm_time / 100  # per sample
    }
    
    # 2. GRU
    print("\n" + "="*50)
    print("Training GRU Predictor")
    print("="*50)
    
    gru = build_gru_predictor(input_length=50, output_length=10)
    
    history_gru = gru.fit(
        data['X_train'], data['y_train'],
        validation_data=(data['X_val'], data['y_val']),
        epochs=100,
        batch_size=32,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
        ]
    )
    
    # Evaluate
    test_loss, test_mae = gru.evaluate(data['X_test'], data['y_test'], verbose=0)
    
    print(f"\nGRU Test Loss (MSE): {test_loss:.6f}")
    print(f"GRU Test MAE: {test_mae:.6f}")
    
    # Timing
    start = time.time()
    predictions_gru = gru.predict(data['X_test'][:100], verbose=0)
    gru_time = time.time() - start
    
    results['GRU'] = {
        'model': gru,
        'history': history_gru.history,
        'test_loss': test_loss,
        'test_mae': test_mae,
        'predictions': predictions_gru,
        'inference_time': gru_time / 100
    }
    
    # 3. Compare with numerical integration time
    print("\n" + "="*50)
    print("Comparing with Numerical Integration")
    print("="*50)
    
    from data_generation import ThreeBodyDataGenerator
    generator = ThreeBodyDataGenerator()
    
    # Time numerical integration (simplified for comparison)
    # We'll just time 10 trajectories to get an estimate
    start = time.time()
    for i in range(10):
        initial_state_scaled = data['X_test'][i, -1, :]  # Last point of input
        # Denormalize
        initial_state = data['scaler'].inverse_transform(initial_state_scaled.reshape(1, -1))[0]
        generator.generate_single_trajectory(initial_state, mu=0.3, t_max=1, n_points=10)
    numerical_time = (time.time() - start) / 10
    
    print(f"\nInference Time Comparison (per sample):")
    print(f"LSTM: {results['LSTM']['inference_time']*1000:.3f} ms")
    print(f"GRU: {results['GRU']['inference_time']*1000:.3f} ms")
    print(f"Numerical Integration: {numerical_time*1000:.3f} ms")
    print(f"\nSpeedup:")
    print(f"LSTM: {numerical_time/results['LSTM']['inference_time']:.1f}x faster")
    print(f"GRU: {numerical_time/results['GRU']['inference_time']:.1f}x faster")
    
    # 4. Plot training history
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
    
    # 5. Visualize example predictions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for idx in range(3):
        # Get one test sample
        X_sample = data['X_test'][idx].copy()  # Shape: [50, 4]
        y_true = data['y_test'][idx].copy()     # Shape: [10, 4]
        
        # Predictions
        y_pred_lstm = results['LSTM']['model'].predict(X_sample[np.newaxis, ...], verbose=0)[0]
        y_pred_gru = results['GRU']['model'].predict(X_sample[np.newaxis, ...], verbose=0)[0]
        
        # Inverse transform (denormalize)
        scaler = data['scaler']
        X_sample = scaler.inverse_transform(X_sample)
        y_true = scaler.inverse_transform(y_true)
        y_pred_lstm = scaler.inverse_transform(y_pred_lstm)
        y_pred_gru = scaler.inverse_transform(y_pred_gru)
        
        # Plot position trajectory
        axes[0, idx].plot(X_sample[:, 0], X_sample[:, 1], 'k-', label='Input', linewidth=2)
        axes[0, idx].plot(y_true[:, 0], y_true[:, 1], 'g-', label='True', linewidth=2, marker='o')
        axes[0, idx].plot(y_pred_lstm[:, 0], y_pred_lstm[:, 1], 'b--', label='LSTM', linewidth=2, marker='s')
        axes[0, idx].plot(y_pred_gru[:, 0], y_pred_gru[:, 1], 'r:', label='GRU', linewidth=2, marker='^')
        axes[0, idx].set_xlabel('ξ')
        axes[0, idx].set_ylabel('η')
        axes[0, idx].set_title(f'Sample {idx+1} - Position')
        axes[0, idx].legend()
        axes[0, idx].grid(True, alpha=0.3)
        axes[0, idx].axis('equal')
        
        # Plot time series for ξ coordinate
        time_input = np.arange(50)
        time_output = np.arange(50, 60)
        
        axes[1, idx].plot(time_input, X_sample[:, 0], 'k-', label='Input', linewidth=2)
        axes[1, idx].plot(time_output, y_true[:, 0], 'g-', label='True', linewidth=2, marker='o')
        axes[1, idx].plot(time_output, y_pred_lstm[:, 0], 'b--', label='LSTM', linewidth=2, marker='s')
        axes[1, idx].plot(time_output, y_pred_gru[:, 0], 'r:', label='GRU', linewidth=2, marker='^')
        axes[1, idx].axvline(x=50, color='gray', linestyle='--', alpha=0.5)
        axes[1, idx].set_xlabel('Time Step')
        axes[1, idx].set_ylabel('ξ')
        axes[1, idx].set_title(f'Sample {idx+1} - ξ Coordinate')
        axes[1, idx].legend()
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
            'inference_time': result['inference_time']
        }
    
    with open('prediction_results.pkl', 'wb') as f:
        pickle.dump(save_results, f)
    
    print("\nResults saved to 'prediction_results.pkl'")
    
    return results

if __name__ == "__main__":
    results = train_prediction_models()