"""
Model Definitions for Three-Body Problem Deep Learning Project
Contains all ML/DL model architectures
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def build_mlp_classifier(input_dim=4, n_classes=4):
    """Simple MLP for classification"""
    model = tf.keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def build_lstm_classifier(input_dim=4, n_classes=4):
    """LSTM for classification (takes sequence input)"""
    model = tf.keras.Sequential([
        layers.Input(shape=(None, input_dim)),  # Variable length sequences
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(32),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def build_lstm_predictor(input_length=50, input_dim=4, output_length=10):
    """LSTM for trajectory prediction"""
    model = tf.keras.Sequential([
        layers.Input(shape=(input_length, input_dim)),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.RepeatVector(output_length),
        layers.LSTM(32, return_sequences=True),
        layers.LSTM(64, return_sequences=True),
        layers.TimeDistributed(layers.Dense(input_dim))
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

def build_gru_predictor(input_length=50, input_dim=4, output_length=10):
    """GRU for trajectory prediction"""
    model = tf.keras.Sequential([
        layers.Input(shape=(input_length, input_dim)),
        layers.GRU(128, return_sequences=True),
        layers.Dropout(0.2),
        layers.GRU(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.GRU(32),
        layers.RepeatVector(output_length),
        layers.GRU(32, return_sequences=True),
        layers.GRU(64, return_sequences=True),
        layers.TimeDistributed(layers.Dense(input_dim))
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

def get_traditional_classifiers():
    """Get traditional ML classifiers"""
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }