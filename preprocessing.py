"""
Data Preprocessing for Three-Body Problem Deep Learning Project
Prepares data for classification and prediction tasks
"""

import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    """Preprocess data for different tasks"""
    
    def __init__(self, dataset_path='three_body_dataset.pkl'):
        with open(dataset_path, 'rb') as f:
            self.dataset = pickle.load(f)
    
    def prepare_classification_data(self):
        """
        Prepare data for RQ2: Stability classification.

        Features (4D initial state):
        - xi:   x-coordinate in rotating frame
        - eta:  y-coordinate in rotating frame
        - vxi:  x-velocity
        - veta: y-velocity

        Labels:
        - 0: Stable
        - 1: Chaotic
        - 2: Escape
        - 3: Collision

        Input shape:  [N, 4]
        Output shape: [N] (integer class labels)

        Returns dict with X_train, X_val, X_test, y_train, y_val, y_test, scaler
        """
        X = []
        y = []

        for data in self.dataset:
            X.append(data['initial_state'])
            y.append(data['label'])

        X = np.array(X)
        y = np.array(y)

        # Split data (stratified to preserve class ratios)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

        # Normalize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)
        X_test  = scaler.transform(X_test)

        print(f"\n{'='*50}")
        print(f"Classification Dataset Statistics")
        print(f"{'='*50}")
        print(f"Total: {len(y)}, Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
        label_names = {0: 'Stable', 1: 'Chaotic', 2: 'Escape', 3: 'Collision'}
        unique, counts = np.unique(y_train, return_counts=True)
        print("Class distribution (train):")
        for label, count in zip(unique, counts):
            print(f"  {label_names[label]}: {count} ({100*count/len(y_train):.1f}%)")
        print(f"{'='*50}\n")

        return {
            'X_train': X_train,
            'X_val':   X_val,
            'X_test':  X_test,
            'y_train': y_train,
            'y_val':   y_val,
            'y_test':  y_test,
            'scaler':  scaler
        }
    
    def prepare_prediction_data(self, input_length=50, output_length=10):
        """
        Prepare data for RQ1: Trajectory prediction.

        Trajectories are split 70/15/15 BEFORE sequence creation to prevent
        data leakage. Sequences are then built within each split independently.

        Input shape:  [N, input_length, 4]  — [xi, eta, vxi, veta] per timestep
        Output shape: [N, output_length, 4]

        Parameters:
        - input_length:  number of past timesteps (default 50)
        - output_length: number of future timesteps to predict (default 10)

        Returns dict with X_train, X_val, X_test, y_train, y_val, y_test, scaler
        """

        # Step 1: Split trajectories first (prevents data leakage)
        train_trajs, temp_trajs = train_test_split(
            self.dataset, test_size=0.3, random_state=42
        )
        val_trajs, test_trajs = train_test_split(
            temp_trajs, test_size=0.5, random_state=42
        )

        # Step 2: Create sliding-window sequences within each split
        def create_sequences(trajectories, input_len, output_len):
            X, y = [], []
            for data in trajectories:
                trajectory = data['trajectory']
                for i in range(len(trajectory) - input_len - output_len):
                    X.append(trajectory[i:i+input_len])
                    y.append(trajectory[i+input_len:i+input_len+output_len])
            return np.array(X), np.array(y)

        X_train, y_train = create_sequences(train_trajs, input_length, output_length)
        X_val,   y_val   = create_sequences(val_trajs,   input_length, output_length)
        X_test,  y_test  = create_sequences(test_trajs,  input_length, output_length)

        # Step 3: Normalize — fit scaler on train only
        n_train, n_val, n_test = X_train.shape[0], X_val.shape[0], X_test.shape[0]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.reshape(-1, 4)).reshape(n_train, input_length, 4)
        X_val   = scaler.transform(X_val.reshape(-1, 4)).reshape(n_val,   input_length, 4)
        X_test  = scaler.transform(X_test.reshape(-1, 4)).reshape(n_test,  input_length, 4)

        y_train = scaler.transform(y_train.reshape(-1, 4)).reshape(n_train, output_length, 4)
        y_val   = scaler.transform(y_val.reshape(-1, 4)).reshape(n_val,   output_length, 4)
        y_test  = scaler.transform(y_test.reshape(-1, 4)).reshape(n_test,  output_length, 4)

        print(f"\n{'='*50}")
        print(f"Prediction Dataset Statistics (No Data Leakage)")
        print(f"{'='*50}")
        print(f"Trajectories — Train: {len(train_trajs)}, Val: {len(val_trajs)}, Test: {len(test_trajs)}")
        print(f"Sequences    — Train: {n_train}, Val: {n_val}, Test: {n_test}")
        print(f"Input shape: {X_train.shape}, Output shape: {y_train.shape}")
        print(f"{'='*50}\n")

        return {
            'X_train': X_train,
            'X_val':   X_val,
            'X_test':  X_test,
            'y_train': y_train,
            'y_val':   y_val,
            'y_test':  y_test,
            'scaler':  scaler
        }
    
    def prepare_equilibrium_data(self):
        """
        Prepare data for RQ3: Discovering Lagrange points
        
        Returns trajectories and their equilibrium proximity
        """
        # Known Lagrange points for mu=0.3 (from thesis Section 3.4)
        mu = 0.3
        L4 = [0.5 - mu, np.sqrt(3)/2]  # (0.2, 0.866)
        L5 = [0.5 - mu, -np.sqrt(3)/2]  # (0.2, -0.866)
        
        lagrange_points = [L4, L5]
        
        equilibrium_data = []
        
        for data in self.dataset:
            if np.abs(data['mu'] - mu) < 0.05:  # Only use mu ≈ 0.3
                trajectory = data['trajectory']
                
                # Find points where velocity is low (near equilibrium)
                velocities = np.sqrt(trajectory[:, 2]**2 + trajectory[:, 3]**2)
                
                # Points with very low velocity
                near_equilibrium_idx = np.where(velocities < 0.1)[0]
                
                for idx in near_equilibrium_idx:
                    pos = trajectory[idx, :2]
                    
                    # Check distance to known Lagrange points
                    min_dist = min([np.linalg.norm(pos - L) for L in lagrange_points])
                    
                    is_near_lagrange = 1 if min_dist < 0.2 else 0
                    
                    equilibrium_data.append({
                        'position': pos,
                        'velocity': trajectory[idx, 2:],
                        'is_equilibrium': is_near_lagrange,
                        'distance_to_nearest_L': min_dist
                    })
        
        return equilibrium_data

# Usage
if __name__ == "__main__":
    preprocessor = DataPreprocessor('three_body_dataset.pkl')
    
    # Classification data
    class_data = preprocessor.prepare_classification_data()
    print(f"Classification - Train: {class_data['X_train'].shape}, Test: {class_data['X_test'].shape}")
    
    # Prediction data
    pred_data = preprocessor.prepare_prediction_data(input_length=50, output_length=10)
    print(f"Prediction - Train: {pred_data['X_train'].shape}, Test: {pred_data['X_test'].shape}")