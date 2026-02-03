"""
Testing Script for Classification Models (RQ2)
Loads saved models and evaluates them on test data
"""

import numpy as np
import pickle
from preprocessing import DataPreprocessor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def test_classification_models():
    """Test saved classification models"""
    
    print("="*70)
    print(" TESTING CLASSIFICATION MODELS (RQ2)")
    print("="*70)
    
    # Load test data
    print("\nLoading test data...")
    preprocessor = DataPreprocessor('three_body_dataset.pkl')
    data = preprocessor.prepare_classification_data()
    
    # Load saved results
    print("Loading saved model results...")
    with open('classification_results.pkl', 'rb') as f:
        results = pickle.load(f)
    
    print("\n" + "="*70)
    print(" TEST RESULTS")
    print("="*70)
    
    class_names = ['Stable', 'Chaotic', 'Escape', 'Collision']
    
    # Display results for each model
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Test Accuracy: {result['test_acc']:.4f}")
        
        # Detailed classification report
        print("\n  Classification Report:")
        print(classification_report(
            data['y_test'], 
            result['predictions'],
            target_names=class_names,
            zero_division=0
        ))
    
    # Compare models
    print("\n" + "="*70)
    print(" MODEL COMPARISON")
    print("="*70)
    
    for name, result in results.items():
        print(f"{name:20s}: {result['test_acc']:.4f}")
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['test_acc'])
    print(f"\n✓ Best Model: {best_model[0]} ({best_model[1]['test_acc']:.4f})")
    
    return results

if __name__ == "__main__":
    test_classification_models()