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
        Prepare data for RQ2: Stability classification
        
        Returns:
        - X: initial conditions [n_samples, 4] (xi, eta, vxi, veta)
        - y: labels [n_samples] (0=stable, 1=chaotic, 2=escape, 3=collision)
        """
        X = []
        y = []
        
        for data in self.dataset:
            X.append(data['initial_state'])
            y.append(data['label'])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        # Normalize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'scaler': scaler
        }
    
    def prepare_prediction_data(self, input_length=50, output_length=10):
        """
        Prepare data for RQ1: Trajectory prediction
        
        Parameters:
        - input_length: number of past timesteps to use
        - output_length: number of future timesteps to predict
        
        Returns sequences for LSTM/GRU training
        """
        X = []  # Input sequences
        y = []  # Target sequences
        
        for data in self.dataset:
            # Only use stable and chaotic trajectories (not escapes/collisions)
            if data['label'] in [0, 1]:
                trajectory = data['trajectory']
                
                # Create sliding windows
                for i in range(len(trajectory) - input_length - output_length):
                    X.append(trajectory[i:i+input_length])
                    y.append(trajectory[i+input_length:i+input_length+output_length])
        
        X = np.array(X)  # Shape: [n_samples, input_length, 4]
        y = np.array(y)  # Shape: [n_samples, output_length, 4]
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
        
        # Normalize
        # Flatten for scaling
        n_train = X_train.shape[0]
        X_train_flat = X_train.reshape(-1, 4)
        scaler = StandardScaler()
        X_train_flat = scaler.fit_transform(X_train_flat)
        X_train = X_train_flat.reshape(n_train, input_length, 4)
        
        # Apply to val and test
        n_val = X_val.shape[0]
        X_val_flat = X_val.reshape(-1, 4)
        X_val_flat = scaler.transform(X_val_flat)
        X_val = X_val_flat.reshape(n_val, input_length, 4)
        
        n_test = X_test.shape[0]
        X_test_flat = X_test.reshape(-1, 4)
        X_test_flat = scaler.transform(X_test_flat)
        X_test = X_test_flat.reshape(n_test, input_length, 4)
        
        # Also scale targets
        y_train_flat = y_train.reshape(-1, 4)
        y_train_flat = scaler.transform(y_train_flat)
        y_train = y_train_flat.reshape(n_train, output_length, 4)
        
        y_val_flat = y_val.reshape(-1, 4)
        y_val_flat = scaler.transform(y_val_flat)
        y_val = y_val_flat.reshape(n_val, output_length, 4)
        
        y_test_flat = y_test.reshape(-1, 4)
        y_test_flat = scaler.transform(y_test_flat)
        y_test = y_test_flat.reshape(n_test, output_length, 4)
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'scaler': scaler
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