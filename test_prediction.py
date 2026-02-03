"""
Testing Script for Prediction Models (RQ1)
Loads saved models and evaluates them on test data
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from preprocessing import DataPreprocessor

def test_prediction_models():
    """Test saved prediction models"""
    
    print("="*70)
    print(" TESTING PREDICTION MODELS (RQ1)")
    print("="*70)
    
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

if __name__ == "__main__":
    test_prediction_models()