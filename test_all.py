"""
Run all tests at once
"""

def main():
    print("\n" + "="*70)
    print(" RUNNING ALL TESTS")
    print("="*70 + "\n")
    
    # Test Classification
    print("\n\n")
    from test_classification import test_classification_models
    test_classification_models()
    
    # Test Prediction
    print("\n\n")
    from test_prediction import test_prediction_models
    test_prediction_models()
    
    # Test Equilibrium Discovery
    print("\n\n")
    from test_equilibria import test_equilibrium_discovery
    test_equilibrium_discovery()
    
    print("\n\n" + "="*70)
    print(" ALL TESTS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()