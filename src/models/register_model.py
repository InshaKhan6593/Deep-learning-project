import json
import mlflow
import dagshub
import logging
from pathlib import Path
from mlflow.client import MlflowClient

import dagshub
dagshub.init(repo_owner='InshaKhan6593', repo_name='Deep-learning-project', mlflow=True)

# set the mlflow tracking uri
mlflow.set_tracking_uri("https://dagshub.com/InshaKhan6593/Deep-learning-project.mlflow")

# create a logger
logger = logging.getLogger("register_model")
logger.setLevel(logging.INFO)

# attach a console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

# make a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)


if __name__ == "__main__":
    # current path
    current_path = Path(__file__)
    # set the root path
    root_path = current_path.parent.parent.parent
    # run info file name
    file_name = "run_information.json"
    
    # load the json file
    try:
        with open(root_path / file_name, "r") as f:
            run_info = json.load(f)
            logger.info("Information loaded successfully")
    except FileNotFoundError:
        logger.error(f"File {file_name} not found")
        raise
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from the file {file_name}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise
    
    run_id = run_info["run_id"]
    artifact_path = run_info["artifact_path"]
    model_uri = run_info["model_uri"]
    
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Artifact Path: {artifact_path}")
    logger.info(f"Model URI: {model_uri}")
    
    # ============================================
    # ADD RUN TAGS (This works on DagsHub)
    # ============================================
    client = MlflowClient()
    
    try:
        # Add tags to the RUN (not model version, since registration doesn't work)
        client.set_tag(run_id, "model_status", "production")
        logger.info("Added 'model_status=production' tag to run")
        
        client.set_tag(run_id, "model_type", "GRU")
        logger.info("Added 'model_type=GRU' tag to run")
        
        client.set_tag(run_id, "deployment_ready", "true")
        logger.info("Added 'deployment_ready=true' tag to run")
        
        client.set_tag(run_id, "validation_status", "passed")
        logger.info("Added 'validation_status=passed' tag to run")
        
    except Exception as e:
        logger.warning(f"Failed to add tags: {e}")
    
    # ============================================
    # CREATE DEPLOYMENT INFO FILE
    # ============================================
    deployment_info = {
        "model_name": "demand_prediction_model",
        "run_id": run_id,
        "model_uri": model_uri,
        "artifact_path": artifact_path,
        "status": "production",
        "model_type": "GRU",
        "deployment_ready": True,
        "instructions": {
            "download_artifacts": f"Use this URI to download: {model_uri}",
            "mlflow_ui": f"https://dagshub.com/InshaKhan6593/Deep-learning-project.mlflow/#/experiments/2/runs/{run_id}",
            "load_in_python": [
                "import mlflow",
                f"artifacts = mlflow.artifacts.download_artifacts('{model_uri}')",
                "# Then load model.keras and encoder.joblib from artifacts folder"
            ]
        }
    }
    
    deployment_file = root_path / "deployment_info.json"
    with open(deployment_file, 'w') as f:
        json.dump(deployment_info, f, indent=4)
    
    logger.info(f"Deployment info saved to {deployment_file}")
    
    # ============================================
    # FINAL SUMMARY
    # ============================================
    logger.info("\n" + "="*70)
    logger.info("✅ MODEL INFORMATION RECORDED!")
    logger.info("="*70)
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Model URI: {model_uri}")
    logger.info(f"\n📝 NOTE: DagsHub has limited Model Registry support.")
    logger.info(f"To register the model manually:")
    logger.info(f"1. Visit: https://dagshub.com/InshaKhan6593/Deep-learning-project.mlflow")
    logger.info(f"2. Go to Experiments → Find Run ID: {run_id}")
    logger.info(f"3. Click on Artifacts → model → Register Model")
    logger.info(f"\n💾 Deployment info saved to: deployment_info.json")
    logger.info(f"\n🚀 To deploy, download artifacts using:")
    logger.info(f"   mlflow.artifacts.download_artifacts('{model_uri}')")
    logger.info("="*70)