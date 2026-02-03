"""
Data Generation for Three-Body Problem Deep Learning Project
Generates trajectories using numerical integration of the restricted circular three-body problem
"""

import numpy as np
from scipy.integrate import odeint
import pickle
from tqdm import tqdm

class ThreeBodyDataGenerator:
    """Generate three-body problem trajectories"""
    
    def __init__(self, G=1.0):
        self.G = G
    
    def restricted_three_body_equations(self, state, t, mu):
        """
        Restricted circular three-body problem equations (Chapter 3 of thesis)
        
        Parameters:
        - state: [xi, eta, vxi, veta]
        - t: time
        - mu: mass parameter
        
        Returns equations of motion (5.10 from thesis)
        """
        xi, eta, vxi, veta = state
        
        # Distances to primaries (eq 5.8)
        rho1 = np.sqrt((xi + mu)**2 + eta**2)
        rho2 = np.sqrt((xi - (1 - mu))**2 + eta**2)
        
        # Avoid division by zero (collision)
        if rho1 < 1e-3 or rho2 < 1e-3:
            return [0, 0, 0, 0]
        
        # Equations of motion (5.10)
        dxi_dt = vxi
        deta_dt = veta
        dvxi_dt = 2*veta + xi - (1-mu)*(xi+mu)/rho1**3 - mu*(xi-(1-mu))/rho2**3
        dveta_dt = -2*vxi + eta - (1-mu)*eta/rho1**3 - mu*eta/rho2**3
        
        return [dxi_dt, deta_dt, dvxi_dt, dveta_dt]
    
    def calculate_jacobi_constant(self, state, mu):
        """Calculate Jacobi constant (eq 5.12 from thesis)"""
        xi, eta, vxi, veta = state
        
        rho1 = np.sqrt((xi + mu)**2 + eta**2)
        rho2 = np.sqrt((xi - (1-mu))**2 + eta**2)
        
        if rho1 < 1e-10 or rho2 < 1e-10:
            return np.nan
        
        Omega = 0.5*(xi**2 + eta**2) + (1-mu)/rho1 + mu/rho2
        C = 2*Omega - (vxi**2 + veta**2)
        
        return C
    
    def classify_trajectory(self, trajectory, initial_C, mu):
        """
        Classify trajectory stability
        
        Returns:
        - 0: stable (bounded, low energy variation)
        - 1: chaotic (bounded, high energy variation)
        - 2: escape (unbounded)
        - 3: collision
        """
        # Check for collision or escape
        max_distance = np.max(np.sqrt(trajectory[:, 0]**2 + trajectory[:, 1]**2))
        min_distance_primary1 = np.min(np.sqrt((trajectory[:, 0] + mu)**2 + trajectory[:, 1]**2))
        min_distance_primary2 = np.min(np.sqrt((trajectory[:, 0] - (1-mu))**2 + trajectory[:, 1]**2))
        
        # Collision
        if min_distance_primary1 < 0.01 or min_distance_primary2 < 0.01:
            return 3
        
        # Escape
        if max_distance > 5.0:
            return 2
        
        # Calculate Jacobi constant variation
        jacobi_constants = []
        for state in trajectory:
            C = self.calculate_jacobi_constant(state, mu)
            if not np.isnan(C):
                jacobi_constants.append(C)
        
        if len(jacobi_constants) < 10:
            return 3  # Too unstable
        
        # Relative variation in Jacobi constant
        C_std = np.std(jacobi_constants)
        C_mean = np.abs(np.mean(jacobi_constants))
        relative_variation = C_std / (C_mean + 1e-10)
        
        # Classify based on variation
        if relative_variation < 0.01:
            return 0  # Stable
        else:
            return 1  # Chaotic
    
    def generate_single_trajectory(self, initial_state, mu, t_max=50, n_points=500):
        """Generate one trajectory"""
        t = np.linspace(0, t_max, n_points)
        
        try:
            trajectory = odeint(
                self.restricted_three_body_equations,
                initial_state,
                t,
                args=(mu,),
                rtol=1e-8,
                atol=1e-10
            )
            
            # Check validity
            if not np.all(np.isfinite(trajectory)):
                return None
            
            if np.max(np.abs(trajectory)) > 100:
                return None
            
            # Calculate initial Jacobi constant
            initial_C = self.calculate_jacobi_constant(initial_state, mu)
            
            # Classify
            label = self.classify_trajectory(trajectory, initial_C, mu)
            
            return {
                'initial_state': initial_state,
                'trajectory': trajectory,
                'time': t,
                'mu': mu,
                'label': label,
                'initial_jacobi': initial_C
            }
            
        except Exception as e:
            return None
    
    def generate_dataset(self, n_trajectories=5000, mu_range=(0.1, 0.4), 
                        t_max=50, n_points=500, filename='three_body_dataset.pkl'):
        """
        Generate complete dataset
        
        Parameters:
        - n_trajectories: number of trajectories to generate
        - mu_range: range of mass parameters
        - t_max: maximum simulation time
        - n_points: number of time points per trajectory
        """
        dataset = []
        
        print(f"Generating {n_trajectories} trajectories...")
        
        for i in tqdm(range(n_trajectories)):
            # Random mass parameter
            mu = np.random.uniform(mu_range[0], mu_range[1])
            
            # Random initial conditions
            # Position: avoid primaries at (-mu, 0) and (1-mu, 0)
            xi0 = np.random.uniform(-1.5, 1.5)
            eta0 = np.random.uniform(-1.5, 1.5)
            
            # Avoid starting too close to primaries
            dist1 = np.sqrt((xi0 + mu)**2 + eta0**2)
            dist2 = np.sqrt((xi0 - (1-mu))**2 + eta0**2)
            
            if dist1 < 0.1 or dist2 < 0.1:
                continue
            
            # Random velocities
            vxi0 = np.random.uniform(-0.8, 0.8)
            veta0 = np.random.uniform(-0.8, 0.8)
            
            initial_state = [xi0, eta0, vxi0, veta0]
            
            # Generate trajectory
            data = self.generate_single_trajectory(initial_state, mu, t_max, n_points)
            
            if data is not None:
                dataset.append(data)
        
        print(f"\nSuccessfully generated {len(dataset)} valid trajectories")
        
        # Print statistics
        labels = [d['label'] for d in dataset]
        print("\nDataset Statistics:")
        print(f"Stable (0): {labels.count(0)} ({100*labels.count(0)/len(labels):.1f}%)")
        print(f"Chaotic (1): {labels.count(1)} ({100*labels.count(1)/len(labels):.1f}%)")
        print(f"Escape (2): {labels.count(2)} ({100*labels.count(2)/len(labels):.1f}%)")
        print(f"Collision (3): {labels.count(3)} ({100*labels.count(3)/len(labels):.1f}%)")
        
        # Save dataset
        with open(filename, 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"\nDataset saved to {filename}")
        
        return dataset

# Usage
if __name__ == "__main__":
    generator = ThreeBodyDataGenerator()
    dataset = generator.generate_dataset(
        n_trajectories=5000,
        mu_range=(0.1, 0.4),
        t_max=50,
        n_points=500,
        filename='three_body_dataset.pkl'
    )