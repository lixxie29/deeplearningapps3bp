"""
Training Script for Classification Task (RQ2)
Trains models to classify trajectory stability from initial conditions
"""

import numpy as np
import pickle
from preprocessing import DataPreprocessor
from models import build_mlp_classifier, get_traditional_classifiers
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

def train_classification_models():
    """Train all classification models"""
    
    # Load data
    print("Loading and preprocessing data...")
    preprocessor = DataPreprocessor('three_body_dataset.pkl')
    data = preprocessor.prepare_classification_data()
    
    results = {}
    
    # 1. Traditional ML models
    print("\n" + "="*50)
    print("Training Traditional ML Models")
    print("="*50)
    
    traditional_models = get_traditional_classifiers()
    
    for name, model in traditional_models.items():
        print(f"\nTraining {name}...")
        model.fit(data['X_train'], data['y_train'])
        
        # Evaluate
        train_acc = model.score(data['X_train'], data['y_train'])
        val_acc = model.score(data['X_val'], data['y_val'])
        test_acc = model.score(data['X_test'], data['y_test'])
        
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Val Accuracy: {val_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        
        # Predictions
        y_pred = model.predict(data['X_test'])
        
        results[name] = {
            'model': model,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'predictions': y_pred
        }
    
    # 2. MLP
    print("\n" + "="*50)
    print("Training MLP")
    print("="*50)
    
    mlp = build_mlp_classifier()
    
    history_mlp = mlp.fit(
        data['X_train'], data['y_train'],
        validation_data=(data['X_val'], data['y_val']),
        epochs=50,
        batch_size=32,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        ]
    )
    
    # Evaluate
    train_loss, train_acc = mlp.evaluate(data['X_train'], data['y_train'], verbose=0)
    val_loss, val_acc = mlp.evaluate(data['X_val'], data['y_val'], verbose=0)
    test_loss, test_acc = mlp.evaluate(data['X_test'], data['y_test'], verbose=0)
    
    print(f"\nMLP Results:")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Val Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    y_pred_mlp = np.argmax(mlp.predict(data['X_test']), axis=1)
    
    results['MLP'] = {
        'model': mlp,
        'history': history_mlp.history,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'predictions': y_pred_mlp
    }
    
    # 3. Print classification reports
    print("\n" + "="*50)
    print("Classification Reports")
    print("="*50)
    
    class_names = ['Stable', 'Chaotic', 'Escape', 'Collision']
    
    for name in results:
        print(f"\n{name}:")
        print(classification_report(
            data['y_test'], 
            results[name]['predictions'],
            target_names=class_names,
            zero_division=0
        ))
    
    # 4. Plot confusion matrices
    fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 4))
    
    if len(results) == 1:
        axes = [axes]
    
    for idx, (name, result) in enumerate(results.items()):
        cm = confusion_matrix(data['y_test'], result['predictions'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=class_names, yticklabels=class_names)
        axes[idx].set_title(f'{name}\nTest Accuracy: {result["test_acc"]:.3f}')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('classification_confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("\nConfusion matrices saved to 'classification_confusion_matrices.png'")
    plt.close()
    
    # 5. Save results
    # Remove models from results before saving (they're too large)
    save_results = {}
    for name, result in results.items():
        save_results[name] = {
            'train_acc': result['train_acc'],
            'val_acc': result['val_acc'],
            'test_acc': result['test_acc'],
            'predictions': result['predictions']
        }
        if 'history' in result:
            save_results[name]['history'] = result['history']
    
    with open('classification_results.pkl', 'wb') as f:
        pickle.dump(save_results, f)
    
    print("\nResults saved to 'classification_results.pkl'")
    
    return results

if __name__ == "__main__":
    results = train_classification_models()