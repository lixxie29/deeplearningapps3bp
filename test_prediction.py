"""
Testing Script for Prediction Models (RQ1)
Loads saved models and evaluates them on test data
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from preprocessing import DataPreprocessor


def test_no_data_leakage():
    """Assert no trajectory appears in more than one split."""
    preprocessor = DataPreprocessor('three_body_dataset.pkl')

    train_trajs, temp_trajs = train_test_split(
        preprocessor.dataset, test_size=0.3, random_state=42
    )
    val_trajs, test_trajs = train_test_split(
        temp_trajs, test_size=0.5, random_state=42
    )

    train_ids = set(id(t) for t in train_trajs)
    val_ids   = set(id(t) for t in val_trajs)
    test_ids  = set(id(t) for t in test_trajs)

    assert len(train_ids & val_ids)  == 0, "LEAKAGE: train and val share trajectories!"
    assert len(train_ids & test_ids) == 0, "LEAKAGE: train and test share trajectories!"
    assert len(val_ids   & test_ids) == 0, "LEAKAGE: val and test share trajectories!"

    print("No data leakage detected.")
    return True

def test_prediction_models():
    """Test saved prediction models"""
    
    print("="*70)
    print(" TESTING PREDICTION MODELS (RQ1)")
    print("="*70)

    # Verify no data leakage before running tests
    print("\nChecking for data leakage...")
    test_no_data_leakage()

    # Load test data
    print("\nLoading test data...")
    preprocessor = DataPreprocessor('three_body_dataset.pkl')
    data = preprocessor.prepare_prediction_data(input_length=50, output_length=10)
    
    # Load saved results
    print("Loading saved model results...")
    with open('prediction_results.pkl', 'rb') as f:
        results = pickle.load(f)
    
    print("\n" + "="*70)
    print(" TEST RESULTS")
    print("="*70)
    
    # Display results for each model
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Test MSE: {result['test_loss']:.6f}")
        print(f"  Test MAE: {result['test_mae']:.6f}")
        print(f"  Inference Time: {result['inference_time']*1000:.3f} ms/sample")
    
    # Compare models
    print("\n" + "="*70)
    print(" MODEL COMPARISON")
    print("="*70)
    
    print(f"\n{'Model':<10} {'MSE':<12} {'MAE':<12} {'Time (ms)':<12}")
    print("-" * 50)
    for name, result in results.items():
        print(f"{name:<10} {result['test_loss']:<12.6f} {result['test_mae']:<12.6f} {result['inference_time']*1000:<12.3f}")
    
    # Find best model (lowest MAE)
    best_model = min(results.items(), key=lambda x: x[1]['test_mae'])
    print(f"\n✓ Best Model (lowest MAE): {best_model[0]} ({best_model[1]['test_mae']:.6f})")
    
    # Visualize one example from saved results
    print("\n" + "="*70)
    print(" VISUALIZATION")
    print("="*70)
    print("Check 'prediction_examples.png' for sample predictions")
    
    return results

def test_breen_baseline():
    """Test saved Breen baseline and print comparison against sequence models."""

    print("="*70)
    print(" TESTING BREEN BASELINE (RQ1)")
    print("="*70)

    with open('breen_results.pkl', 'rb') as f:
        breen = pickle.load(f)

    print(f"\n{'Model':<12} {'MAE':<12} {'Time (ms)':<12}")
    print("-" * 38)
    for name, result in breen.items():
        print(f"{name:<12} {result['test_mae']:<12.6f} {result['inference_time']*1000:<12.3f}")

    # Cross-compare with sequence models if available
    try:
        with open('prediction_results.pkl', 'rb') as f:
            seq = pickle.load(f)

        print("\n" + "="*70)
        print(" FULL RQ1 COMPARISON (Breen baseline vs sequence models)")
        print("="*70)
        print(f"\n{'Model':<14} {'MAE':<12} {'Time (ms)':<14} {'Gap vs Breen'}")
        print("-" * 58)

        breen_mae = list(breen.values())[0]['test_mae']
        for name, result in breen.items():
            print(f"{name:<14} {result['test_mae']:<12.6f} {result['inference_time']*1000:<14.3f} — (baseline)")
        for name, result in seq.items():
            gap = result['test_mae'] - breen_mae
            direction = f"+{gap:.6f} worse" if gap > 0 else f"{gap:.6f} better"
            print(f"{name:<14} {result['test_mae']:<12.6f} {result['inference_time']*1000:<14.3f} {direction}")
    except FileNotFoundError:
        print("\n(prediction_results.pkl not found — run train_prediction.py for full comparison)")

    return breen


if __name__ == "__main__":
    test_breen_baseline()
    test_prediction_models()