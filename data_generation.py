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
    
    def _sample_stable(self, mu):
        """Sample near L4/L5 — the known stable equilibrium points."""
        sign = np.random.choice([-1, 1])
        xi0  = (0.5 - mu) + np.random.uniform(-0.3, 0.3)
        eta0 = sign * np.sqrt(3) / 2 + np.random.uniform(-0.3, 0.3)
        vxi0  = np.random.uniform(-0.15, 0.15)
        veta0 = np.random.uniform(-0.15, 0.15)
        return xi0, eta0, vxi0, veta0

    def _sample_chaotic(self, mu):
        """Sample near the L1 unstable equilibrium — the stability boundary."""
        r_hill = (mu / 3) ** (1 / 3)
        L1_x = (1 - mu) - r_hill
        xi0  = L1_x + np.random.uniform(-0.2, 0.2)
        eta0 = np.random.uniform(-0.3, 0.3)
        vxi0  = np.random.uniform(-0.3, 0.3)
        veta0 = np.random.uniform(-0.3, 0.3)
        return xi0, eta0, vxi0, veta0

    def _sample_collision(self, mu):
        """Sample close to one of the primaries to encourage collision trajectories."""
        primary_x = np.random.choice([-mu, 1 - mu])
        angle = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0.02, 0.12)
        xi0  = primary_x + r * np.cos(angle)
        eta0 = r * np.sin(angle)
        vxi0  = np.random.uniform(-0.5, 0.5)
        veta0 = np.random.uniform(-0.5, 0.5)
        return xi0, eta0, vxi0, veta0

    def generate_balanced_dataset(self, target_counts=None, mu_range=(0.1, 0.4),
                                   t_max=50, n_points=500, filename='three_body_dataset.pkl',
                                   max_attempts=100000):
        """
        Generate a dataset with explicit per-class quotas using targeted sampling.

        Pure random sampling gives ~73% Escape, ~20% Stable, ~6% Collision, ~0.1% Chaotic,
        making the Chaotic class practically unlearnable. This method biases sampling toward
        whichever class is most behind its quota:
          - Stable:    sample near L4/L5 equilibrium points
          - Chaotic:   sample near the L1 unstable equilibrium (stability boundary)
          - Collision: sample close to a primary body
          - Escape:    pure random (happens naturally)

        Parameters
        ----------
        target_counts : dict, optional
            {label: count} targets. Default gives a roughly balanced 4-class dataset.
        max_attempts : int
            Hard cap on integration attempts to prevent infinite loops.
        """
        if target_counts is None:
            target_counts = {0: 1200, 1: 500, 2: 1200, 3: 600}

        counts  = {k: 0 for k in target_counts}
        dataset = []
        attempts = 0
        total_target = sum(target_counts.values())

        label_names = {0: 'Stable', 1: 'Chaotic', 2: 'Escape', 3: 'Collision'}
        print(f"Generating balanced dataset — targets: "
              f"{ {label_names[k]: v for k, v in target_counts.items()} }")

        pbar = tqdm(total=total_target, desc='Trajectories collected')

        while attempts < max_attempts:
            # Stop when every class has hit its quota
            if all(counts[k] >= target_counts[k] for k in target_counts):
                break

            attempts += 1
            mu = np.random.uniform(mu_range[0], mu_range[1])

            # Pick the most-needed class and use its targeted sampler
            needed = [k for k in target_counts if counts[k] < target_counts[k]]
            target_class = np.random.choice(needed)

            if target_class == 0:
                xi0, eta0, vxi0, veta0 = self._sample_stable(mu)
            elif target_class == 1:
                xi0, eta0, vxi0, veta0 = self._sample_chaotic(mu)
            elif target_class == 3:
                xi0, eta0, vxi0, veta0 = self._sample_collision(mu)
            else:
                xi0   = np.random.uniform(-1.5, 1.5)
                eta0  = np.random.uniform(-1.5, 1.5)
                vxi0  = np.random.uniform(-0.8, 0.8)
                veta0 = np.random.uniform(-0.8, 0.8)

            dist1 = np.sqrt((xi0 + mu)**2 + eta0**2)
            dist2 = np.sqrt((xi0 - (1 - mu))**2 + eta0**2)
            if target_class != 3 and (dist1 < 0.05 or dist2 < 0.05):
                continue

            data = self.generate_single_trajectory(
                [xi0, eta0, vxi0, veta0], mu, t_max, n_points
            )
            if data is None:
                continue

            label = data['label']
            if counts.get(label, 0) < target_counts.get(label, 0):
                dataset.append(data)
                counts[label] += 1
                pbar.update(1)

        pbar.close()

        print(f"\nCollected {len(dataset)} trajectories in {attempts} attempts")
        print("Final class distribution:")
        for k, name in label_names.items():
            n = counts.get(k, 0)
            pct = 100 * n / len(dataset) if dataset else 0
            print(f"  {name}: {n} ({pct:.1f}%)")

        with open(filename, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"Dataset saved to {filename}")
        return dataset

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