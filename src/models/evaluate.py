import mlflow
import dagshub
import json
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging
from sklearn import set_config
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow import keras
import tempfile
import shutil

import dagshub
dagshub.init(repo_owner='InshaKhan6593', repo_name='Deep-learning-project', mlflow=True)

# set the mlflow tracking uri
mlflow.set_tracking_uri("https://dagshub.com/InshaKhan6593/Deep-learning-project.mlflow")

# set the experiment name
mlflow.set_experiment("DVC Pipeline")

set_config(transform_output="pandas")

# create a logger
logger = logging.getLogger("evaluate_model")
logger.setLevel(logging.INFO)

# attach a console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

# make a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

def load_keras_model(model_path):
    """Load Keras model"""
    model = keras.models.load_model(model_path)
    return model

def save_run_information(run_id, artifact_path, model_uri, path):
    run_information = {
        "run_id": run_id,
        "artifact_path": artifact_path,
        "model_uri": model_uri
    }
    with open(path, "w") as f:
        json.dump(run_information, f, indent=4)


if __name__ == "__main__":
    # current path
    current_path = Path(__file__)
    # set the root path
    root_path = current_path.parent.parent.parent
    # data_path
    train_data_path = root_path / "data/processed/train.csv"
    test_data_path = root_path / "data/processed/test.csv"
    
    # read the data
    df = pd.read_csv(test_data_path, parse_dates=["tpep_pickup_datetime"])
    logger.info("Data read successfully")
    
    # set the datetime column as index
    df.set_index("tpep_pickup_datetime", inplace=True)
    
    # make X_test and y_test
    X_test = df.drop(columns=["total_pickups"])
    y_test = df["total_pickups"].values  # Convert to numpy array
    
    # load the encoder
    encoder_path = root_path / "models/encoder.joblib"
    encoder = joblib.load(encoder_path)
    logger.info("Encoder loaded successfully")
    
    # transform the test data
    X_test_encoded = encoder.transform(X_test)
    logger.info("Data transformed successfully")
    
    # Convert to numpy if DataFrame
    if isinstance(X_test_encoded, pd.DataFrame):
        X_test_encoded = X_test_encoded.values
    
    # Reshape for GRU model (samples, timesteps, features)
    total_features = X_test_encoded.shape[1]
    X_test_reshaped = X_test_encoded.reshape(X_test_encoded.shape[0], 1, total_features)
    logger.info(f"Data reshaped for GRU: {X_test_reshaped.shape}")
    
    # Load Keras model
    model_path = root_path / "models/model.keras"
    model = load_keras_model(model_path)
    logger.info("Model loaded successfully")
    
    # make predictions
    y_pred = model.predict(X_test_reshaped, verbose=0).flatten()
    logger.info(f"Predictions made: {y_pred.shape}")
    
    # calculate the loss
    loss = mean_absolute_percentage_error(y_test, y_pred)
    logger.info(f"MAPE: {loss:.4f}")
    
    # Load model metadata for parameters
    metadata_path = root_path / "models/model_metadata.json"
    with open(metadata_path, 'r') as f:
        model_params = json.load(f)
    
    # mlflow tracking
    with mlflow.start_run(run_name="model") as run:    
        # Log the model parameters from metadata
        mlflow.log_params({
            "model_name": model_params['model_name'],
            "units": model_params['units'],
            "learning_rate": model_params['learning_rate'],
            "batch_size": model_params['batch_size'],
            "epochs_trained": model_params['epochs_trained']
        })
        
        # log the metrics
        mlflow.log_metric("test_MAPE", loss)
        mlflow.log_metric("train_loss", model_params['final_train_loss'])
        mlflow.log_metric("val_loss", model_params['final_val_loss'])
        mlflow.log_metric("train_mae", model_params['final_train_mae'])
        mlflow.log_metric("val_mae", model_params['final_val_mae'])
        
        # converts the datasets into mlflow datasets
        training_data = mlflow.data.from_pandas(
            pd.read_csv(train_data_path, parse_dates=["tpep_pickup_datetime"]).set_index("tpep_pickup_datetime"), 
            targets="total_pickups"
        )
        
        validation_data = mlflow.data.from_pandas(
            pd.read_csv(test_data_path, parse_dates=["tpep_pickup_datetime"]).set_index("tpep_pickup_datetime"), 
            targets="total_pickups"
        )
        
        # log the datasets
        mlflow.log_input(training_data, "training")
        mlflow.log_input(validation_data, "validation")
        
        # ============================================
        # LOG MODEL AS ARTIFACT (DagsHub Compatible)
        # ============================================
        # Create a temporary directory to save model files
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_model_path = Path(tmp_dir) / "model.keras"
            tmp_encoder_path = Path(tmp_dir) / "encoder.joblib"
            tmp_metadata_path = Path(tmp_dir) / "model_metadata.json"
            
            # Copy model files to temp directory
            shutil.copy(model_path, tmp_model_path)
            shutil.copy(encoder_path, tmp_encoder_path)
            shutil.copy(metadata_path, tmp_metadata_path)
            
            # Log all artifacts
            mlflow.log_artifact(str(tmp_model_path), artifact_path="model")
            mlflow.log_artifact(str(tmp_encoder_path), artifact_path="model")
            mlflow.log_artifact(str(tmp_metadata_path), artifact_path="model")
            
            logger.info("Model artifacts logged successfully")
        
        # Get run information
        run_id = run.info.run_id
        artifact_path = "model"
        model_uri = f"runs:/{run_id}/{artifact_path}"
        
        logger.info("MLflow logging complete")
    
    # save to json file
    json_file_save_path = root_path / "run_information.json"
    save_run_information(
        run_id=run_id,
        artifact_path=artifact_path,
        model_uri=model_uri,
        path=json_file_save_path
    )
    logger.info("Run information saved successfully")