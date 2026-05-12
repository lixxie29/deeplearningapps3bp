"""
Complete workflow for Three-Body Problem Deep Learning Project
Run this to execute all experiments
"""

import sys
import os
from s3_utils import download_dataset, upload_all_results

def main():
    print("="*70)
    print(" THREE-BODY PROBLEM DEEP LEARNING PROJECT")
    print("="*70)
    
    # Step 1: Generate data
    print("\n\nSTEP 1: Generating Dataset")
    print("-"*70)
    
    if not os.path.exists('three_body_dataset.pkl'):
        download_dataset()  # try S3 first

    if not os.path.exists('three_body_dataset.pkl'):
        from data_generation import ThreeBodyDataGenerator
        generator = ThreeBodyDataGenerator()
        dataset = generator.generate_balanced_dataset(
            target_counts={0: 30000, 1: 15000, 2: 30000, 3: 15000},
            mu_range=(0.1, 0.4),
            t_max=50,
            n_points=500,
            filename='three_body_dataset.pkl'
        )
        print("✓ Dataset generated successfully")
    else:
        print("✓ Dataset already exists, skipping generation")
    
    # Step 2: Classification (RQ2)
    print("\n\nSTEP 2: Training Classification Models (RQ2)")
    print("-"*70)
    
    from train_classification import train_classification_models
    class_results = train_classification_models()
    print("✓ Classification models trained successfully")
    
    # Step 3: Breen Baseline (RQ1 — train first as the reference ceiling)
    print("\n\nSTEP 3: Training Breen Baseline (RQ1)")
    print("-"*70)

    from train_breen_baseline import train_breen_baseline
    breen_results = train_breen_baseline()
    print("✓ Breen baseline trained successfully")

    # Step 4: Prediction models (RQ1 — compared against Breen)
    print("\n\nSTEP 4: Training Prediction Models (RQ1)")
    print("-"*70)

    from train_prediction import train_prediction_models
    pred_results = train_prediction_models()
    print("✓ Prediction models trained successfully")

    # Step 5: Equilibrium Discovery (RQ3)
    print("\n\nSTEP 5: Discovering Lagrange Points (RQ3)")
    print("-"*70)
    
    from discover_equilibria import discover_lagrange_points
    eq_results = discover_lagrange_points()
    print("✓ Equilibrium discovery completed successfully")
    
    # Summary
    print("\n\n" + "="*70)
    print(" EXPERIMENT COMPLETE")
    print("="*70)
    
    print("\nGenerated Files:")
    print("  - three_body_dataset.pkl")
    print("  - classification_results.pkl")
    print("  - classification_confusion_matrices.png")
    print("  - prediction_results.pkl")
    print("  - prediction_training_history.png")
    print("  - prediction_examples.png")
    print("  - breen_results.pkl")
    print("  - breen_training_history.png")
    print("  - breen_prediction_examples.png")
    print("  - equilibrium_discovery_results.pkl")
    print("  - lagrange_point_discovery.png")

    print("\nResults Summary:")
    print("\nRQ2 - Classification:")
    for name, result in class_results.items():
        print(f"  {name}: {result['test_acc']:.3f} accuracy")

    print("\nRQ1 - Prediction (sequence models):")
    for name, result in pred_results.items():
        print(f"  {name}: {result['test_mae']:.6f} MAE, {result['inference_time']*1000:.2f} ms/sample")

    print("\nRQ1 - Breen Baseline:")
    for name, result in breen_results.items():
        print(f"  {name}: {result['test_mae']:.6f} MAE, {result['inference_time']*1000:.2f} ms/sample")

    print("\nRQ3 - Equilibrium Discovery:")
    print(f"  Discovered {len(eq_results['discovered_points'])} equilibrium regions")
    
    print("\n" + "="*70)

    # Upload all results to S3
    upload_all_results()

if __name__ == "__main__":
    main()