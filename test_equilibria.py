"""
Testing Script for Equilibrium Discovery (RQ3)
Loads and analyzes discovered Lagrange points
"""

import numpy as np
import pickle

def test_equilibrium_discovery():
    """Test equilibrium discovery results"""
    
    print("="*70)
    print(" TESTING EQUILIBRIUM DISCOVERY (RQ3)")
    print("="*70)
    
    # Load saved results
    print("\nLoading saved results...")
    with open('equilibrium_discovery_results.pkl', 'rb') as f:
        results = pickle.load(f)
    
    discovered = results['discovered_points']
    known = results['known_points']
    
    print("\n" + "="*70)
    print(" RESULTS")
    print("="*70)
    
    print(f"\nDiscovered {len(discovered)} equilibrium regions")
    
    print("\nKnown Lagrange Points (μ=0.3):")
    print(f"  L4: ({known[0][0]:.3f}, {known[0][1]:.3f})")
    print(f"  L5: ({known[1][0]:.3f}, {known[1][1]:.3f})")
    
    if len(discovered) > 0:
        print("\nDiscovered Points:")
        from scipy.spatial.distance import cdist
        distances = cdist(discovered, known)
        
        for i, point in enumerate(discovered):
            nearest_L = np.argmin(distances[i])
            min_dist = distances[i, nearest_L]
            print(f"  Point {i}: ({point[0]:.3f}, {point[1]:.3f}) - Distance to L{4+nearest_L}: {min_dist:.3f}")
        
        # Calculate average distance
        avg_dist = np.mean([np.min(distances[i]) for i in range(len(discovered))])
        print(f"\n✓ Average Distance to Nearest Lagrange Point: {avg_dist:.3f}")
        
        if avg_dist < 0.2:
            print("✓ SUCCESS: Discovered points are close to known Lagrange points!")
        else:
            print("⚠ WARNING: Discovered points are far from known Lagrange points")
    else:
        print("\n⚠ No equilibrium points discovered")
    
    print("\n" + "="*70)
    print(" VISUALIZATION")
    print("="*70)
    print("Check 'lagrange_point_discovery.png' for detailed visualization")
    
    return results

if __name__ == "__main__":
    test_equilibrium_discovery()