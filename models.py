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

def build_transformer_predictor(input_length=50, input_dim=4, output_length=10,
                                d_model=64, num_heads=4, num_layers=2, dff=128, dropout_rate=0.1):
    """
    Transformer encoder-decoder for trajectory prediction.

    Applies multi-head self-attention across time steps, allowing the model to
    directly relate any two timesteps in the input window regardless of distance.
    Unlike LSTM/GRU which process sequentially, attention is computed in parallel
    over all 50 input timesteps simultaneously.

    Architecture:
    - Linear projection to d_model
    - Sinusoidal positional encoding (injects time order information)
    - num_layers x Transformer encoder blocks (self-attention + FFN)
    - Global average pooling → RepeatVector → TimeDistributed Dense output
    """
    inputs = layers.Input(shape=(input_length, input_dim))

    # Project input features to d_model dimensions
    x = layers.Dense(d_model)(inputs)

    # Sinusoidal positional encoding
    positions = tf.cast(tf.range(input_length), tf.float32)
    dims = tf.cast(tf.range(d_model), tf.float32)
    angles = positions[:, tf.newaxis] / tf.pow(10000.0, (2 * (dims[tf.newaxis, :] // 2)) / tf.cast(d_model, tf.float32))
    sin_enc = tf.math.sin(angles[:, 0::2])
    cos_enc = tf.math.cos(angles[:, 1::2])
    pos_encoding = tf.concat([sin_enc, cos_enc], axis=-1)[:, :d_model]
    x = x + pos_encoding

    # Transformer encoder blocks
    for _ in range(num_layers):
        # Multi-head self-attention
        attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(x, x)
        attn_output = layers.Dropout(dropout_rate)(attn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

        # Position-wise feed-forward network
        ffn = layers.Dense(dff, activation='relu')(x)
        ffn = layers.Dense(d_model)(ffn)
        ffn = layers.Dropout(dropout_rate)(ffn)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ffn)

    # Pool across time and decode to output sequence
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.RepeatVector(output_length)(x)
    x = layers.TimeDistributed(layers.Dense(dff, activation='relu'))(x)
    outputs = layers.TimeDistributed(layers.Dense(input_dim))(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def build_breen_mlp(input_dim=6, output_dim=4, hidden_layers=10, hidden_units=128):
    """
    Breen et al. (2019) baseline adapted for the restricted circular 3BP.

    Maps (t, xi_0, eta_0, vxi_0, veta_0, mu) → (xi(t), eta(t), vxi(t), veta(t))
    directly from initial conditions and a query time, with no sequence context.

    Original paper used input_dim=3 (equal-mass, zero-velocity, 2D free 3BP).
    Here input_dim=6 to cover the full 4D IC vector plus mass parameter mu.

    Architecture: 10 x Dense(128, ReLU) + Dense(4, linear)
    Loss: MAE  |  Optimizer: Adam  (matches paper exactly)
    """
    model = tf.keras.Sequential(
        [layers.Input(shape=(input_dim,))]
        + [layers.Dense(hidden_units, activation='relu') for _ in range(hidden_layers)]
        + [layers.Dense(output_dim)]
    )
    model.compile(optimizer='adam', loss='mae', metrics=['mae'])
    return model


def get_traditional_classifiers():
    """Get traditional ML classifiers"""
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    }