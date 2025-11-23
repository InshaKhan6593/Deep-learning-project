import pandas as pd
import numpy as np
import joblib
import logging
import yaml
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn import set_config
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import json

set_config(transform_output="pandas")

# create a logger
logger = logging.getLogger("train_model")
logger.setLevel(logging.INFO)

# attach a console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

# make a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

def load_params(params_path):
    """Load parameters from params.yaml"""
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)
    return params

def save_model(model, save_path):
    """Save Keras model"""
    # Ensure the path has .keras extension (recommended for Keras 3.x)
    if not str(save_path).endswith('.keras'):
        save_path = str(save_path).replace('.joblib', '.keras').replace('.h5', '.keras')
    
    model.save(save_path)
    logger.info(f"Model saved to {save_path}")

def build_gru_model(input_shape, units, learning_rate):
    """
    Build GRU model with parameters from params.yaml
    """
    model = keras.Sequential([
        layers.GRU(units=units, return_sequences=True, input_shape=input_shape),
        layers.Dropout(0.2),
        layers.GRU(units=units//2, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    
    # Compile with learning rate from params
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

if __name__ == "__main__":
    # current path
    current_path = Path(__file__)
    # set the root path
    root_path = current_path.parent.parent.parent
    
    # Load parameters from params.yaml
    params_path = root_path / "params.yaml"
    params = load_params(params_path)
    
    # Extract model parameters
    model_name = params['train']['model_name']
    units = params['train']['units']
    learning_rate = params['train']['learning_rate']
    batch_size = params['train'].get('batch_size', 64)
    epochs = params['train'].get('epochs', 100)
    
    logger.info(f"Loaded parameters from {params_path}")
    logger.info(f"Model: {model_name}, Units: {units}, LR: {learning_rate}")
    
    # data_path
    data_path = root_path / "data/processed/train.csv"
    
    # read the data
    df = pd.read_csv(data_path, parse_dates=["tpep_pickup_datetime"])
    logger.info("Data read successfully")
    
    # set the datetime column as index
    df.set_index("tpep_pickup_datetime", inplace=True)
    
    # make X_train and y_train
    X_train = df.drop(columns=["total_pickups"])
    y_train = df["total_pickups"].values
    logger.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    
    # make the transformer
    encoder = ColumnTransformer([
        ("ohe", OneHotEncoder(drop="first", sparse_output=False), ["region", "day_of_week"])
    ], remainder="passthrough", n_jobs=-1, force_int_remainder_cols=False)
    
    # fit the transformer
    encoder.fit(X_train)
    
    # save the transformer
    encoder_save_path = root_path / "models/encoder.joblib"
    joblib.dump(encoder, encoder_save_path)
    logger.info(f"Encoder saved to {encoder_save_path}")
    
    # encode the training data
    X_train_encoded = encoder.fit_transform(X_train)
    logger.info("Data encoded successfully")
    
    # Convert to numpy if it's a DataFrame
    if isinstance(X_train_encoded, pd.DataFrame):
        X_train_encoded = X_train_encoded.values
    
    # Reshape data for GRU: (samples, timesteps, features)
    total_features = X_train_encoded.shape[1]
    X_train_reshaped = X_train_encoded.reshape(X_train_encoded.shape[0], 1, total_features)
    logger.info(f"Data reshaped for GRU: {X_train_reshaped.shape}")
    
    # Build the model with params from params.yaml
    input_shape = (1, total_features)
    model = build_gru_model(
        input_shape=input_shape,
        units=units,
        learning_rate=learning_rate
    )
    
    # Print model summary
    logger.info("Model architecture:")
    model.summary(print_fn=lambda x: logger.info(x))
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-7,
        verbose=1
    )
    
    # Train the model
    logger.info("Starting model training...")
    history = model.fit(
        X_train_reshaped, 
        y_train,
        validation_split=0.2,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    logger.info(f"Model trained successfully in {len(history.history['loss'])} epochs")
    
    # Log final metrics
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_train_mae = history.history['mae'][-1]
    final_val_mae = history.history['val_mae'][-1]
    
    logger.info(f"Final Training Loss: {final_train_loss:.4f}")
    logger.info(f"Final Validation Loss: {final_val_loss:.4f}")
    logger.info(f"Final Training MAE: {final_train_mae:.4f}")
    logger.info(f"Final Validation MAE: {final_val_mae:.4f}")
    
    # ✅ Save the model with .keras extension (Keras 3.x recommended format)
    model_save_path = root_path / "models/model.keras"
    save_model(model, model_save_path)
    
    # Save training history
    history_save_path = root_path / "models/training_history.json"
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(history_save_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    logger.info(f"Training history saved to {history_save_path}")
    
    # Save model metadata
    metadata = {
        'model_name': model_name,
        'units': units,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epochs_trained': len(history.history['loss']),
        'final_train_loss': float(final_train_loss),
        'final_val_loss': float(final_val_loss),
        'final_train_mae': float(final_train_mae),
        'final_val_mae': float(final_val_mae)
    }
    
    metadata_save_path = root_path / "models/model_metadata.json"
    with open(metadata_save_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Model metadata saved to {metadata_save_path}")
    
    logger.info("✅ Training pipeline completed successfully!")