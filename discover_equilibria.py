"""
Equilibrium Discovery Script (RQ3)
Discovers Lagrange points from trajectory data using clustering
"""

import numpy as np
import pickle
from preprocessing import DataPreprocessor
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist

def discover_lagrange_points():
    """
    Discover Lagrange points from trajectory data
    Uses clustering on low-velocity regions
    """
    
    print("Loading data...")
    preprocessor = DataPreprocessor('three_body_dataset.pkl')
    
    # Collect all low-velocity points
    low_velocity_points = []
    mu_values = []
    
    for data in preprocessor.dataset:
        mu = data['mu']
        trajectory = data['trajectory']
        
        # Only use stable trajectories near mu=0.3
        if data['label'] == 0 and 0.25 < mu < 0.35:
            velocities = np.sqrt(trajectory[:, 2]**2 + trajectory[:, 3]**2)
            
            # Find points with very low velocity
            low_vel_idx = np.where(velocities < 0.05)[0]
            
            for idx in low_vel_idx:
                low_velocity_points.append(trajectory[idx, :2])  # Only position
                mu_values.append(mu)
    
    low_velocity_points = np.array(low_velocity_points)
    mu_values = np.array(mu_values)
    
    print(f"Found {len(low_velocity_points)} low-velocity points")
    
    # Cluster to find equilibrium regions
    print("\nClustering low-velocity points...")
    clustering = DBSCAN(eps=0.15, min_samples=10)
    labels = clustering.fit_predict(low_velocity_points)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Found {n_clusters} clusters")
    
    # Find cluster centers
    discovered_points = []
    
    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        cluster_points = low_velocity_points[cluster_mask]
        
        # Center of cluster
        center = np.mean(cluster_points, axis=0)
        discovered_points.append(center)
        
        print(f"\nCluster {cluster_id}:")
        print(f"  Center: ({center[0]:.3f}, {center[1]:.3f})")
        print(f"  Size: {len(cluster_points)} points")
    
    discovered_points = np.array(discovered_points)
    
    # Known Lagrange points for mu=0.3
    mu = 0.3
    L4 = [0.5 - mu, np.sqrt(3)/2]
    L5 = [0.5 - mu, -np.sqrt(3)/2]
    
    known_lagrange = np.array([L4, L5])
    
    print("\n" + "="*50)
    print("Comparison with Known Lagrange Points")
    print("="*50)
    
    print("\nKnown L4:", L4)
    print("Known L5:", L5)
    
    # Match discovered points to known points
    if len(discovered_points) > 0:
        distances = cdist(discovered_points, known_lagrange)
        
        for i, point in enumerate(discovered_points):
            nearest_L = np.argmin(distances[i])
            min_dist = distances[i, nearest_L]
            
            print(f"\nDiscovered point {i}: ({point[0]:.3f}, {point[1]:.3f})")
            print(f"  Nearest Lagrange point: L{4+nearest_L}")
            print(f"  Distance: {min_dist:.3f}")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: All low-velocity points with clusters
    scatter = axes[0].scatter(
        low_velocity_points[:, 0], 
        low_velocity_points[:, 1],
        c=labels,
        cmap='viridis',
        alpha=0.5,
        s=10
    )
    
    # Plot discovered centers
    if len(discovered_points) > 0:
        axes[0].scatter(
            discovered_points[:, 0],
            discovered_points[:, 1],
            c='red',
            marker='*',
            s=500,
            edgecolors='black',
            linewidths=2,
            label='Discovered Equilibria',
            zorder=5
        )
    
    # Plot known Lagrange points
    axes[0].scatter(
        known_lagrange[:, 0],
        known_lagrange[:, 1],
        c='blue',
        marker='X',
        s=500,
        edgecolors='black',
        linewidths=2,
        label='Known L4, L5',
        zorder=5
    )
    
    # Plot primaries
    axes[0].scatter([-mu], [0], c='orange', marker='o', s=200, label='Sun', edgecolors='black', zorder=5)
    axes[0].scatter([1-mu], [0], c='cyan', marker='o', s=100, label='Planet', edgecolors='black', zorder=5)
    
    axes[0].set_xlabel('ξ')
    axes[0].set_ylabel('η')
    axes[0].set_title('Discovered vs Known Lagrange Points')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axis('equal')
    
    # Plot 2: Heatmap of low-velocity point density
    from scipy.stats import gaussian_kde
    
    if len(low_velocity_points) > 100:
        # Create density estimate
        xy = low_velocity_points.T
        kde = gaussian_kde(xy)
        
        # Create grid
        xi_range = np.linspace(-1.5, 1.5, 100)
        eta_range = np.linspace(-1.5, 1.5, 100)
        Xi, Eta = np.meshgrid(xi_range, eta_range)
        positions = np.vstack([Xi.ravel(), Eta.ravel()])
        
        # Evaluate density
        Z = kde(positions).reshape(Xi.shape)
        
        # Plot heatmap
        im = axes[1].contourf(Xi, Eta, Z, levels=20, cmap='hot')
        plt.colorbar(im, ax=axes[1], label='Density')
        
        # Overlay known Lagrange points
        axes[1].scatter(
            known_lagrange[:, 0],
            known_lagrange[:, 1],
            c='cyan',
            marker='X',
            s=500,
            edgecolors='white',
            linewidths=2,
            label='Known L4, L5',
            zorder=5
        )
        
        if len(discovered_points) > 0:
            axes[1].scatter(
                discovered_points[:, 0],
                discovered_points[:, 1],
                c='lime',
                marker='*',
                s=500,
                edgecolors='white',
                linewidths=2,
                label='Discovered',
                zorder=5
            )
        
        axes[1].scatter([-mu], [0], c='yellow', marker='o', s=200, label='Sun', edgecolors='white', linewidths=2, zorder=5)
        axes[1].scatter([1-mu], [0], c='lightblue', marker='o', s=100, label='Planet', edgecolors='white', linewidths=2, zorder=5)
        
        axes[1].set_xlabel('ξ')
        axes[1].set_ylabel('η')
        axes[1].set_title('Low-Velocity Point Density Heatmap')
        axes[1].legend()
        axes[1].axis('equal')
    
    plt.tight_layout()
    plt.savefig('lagrange_point_discovery.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved to 'lagrange_point_discovery.png'")
    plt.close()
    
    # Save results
    results = {
        'discovered_points': discovered_points,
        'known_points': known_lagrange,
        'low_velocity_points': low_velocity_points,
        'cluster_labels': labels
    }
    
    with open('equilibrium_discovery_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("\nResults saved to 'equilibrium_discovery_results.pkl'")
    
    return results

if __name__ == "__main__":
    results = discover_lagrange_points()