"""
Prefect Deployment Configuration for MLOps Project
Automated retraining pipeline with scheduling
"""

from prefect import flow, task
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule
import subprocess
import sys
from pathlib import Path

@task
def train_model_task():
    """Task to train the ML model"""
    try:
        # Run the training pipeline
        result = subprocess.run([
            sys.executable, "src/modelling/main.py"
        ], capture_output=True, text=True, cwd=Path.cwd())
        
        if result.returncode == 0:
            print("✅ Model training completed successfully")
            return True
        else:
            print(f"❌ Model training failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error during model training: {e}")
        return False

@flow(name="mlops-training-pipeline")
def mlops_training_flow():
    """Main flow for MLOps training pipeline"""
    print("🚀 Starting MLOps training pipeline...")
    
    # Train the model
    success = train_model_task()
    
    if success:
        print("🎉 MLOps training pipeline completed successfully!")
    else:
        print("💥 MLOps training pipeline failed!")
        raise Exception("Model training failed")
    
    return success

# Create deployment
deployment = Deployment.build_from_flow(
    flow=mlops_training_flow,
    name="mlops-training-deployment",
    schedule=CronSchedule(cron="0 2 * * *", timezone="UTC"),  # Daily at 2 AM UTC
    work_pool_name="default-agent-pool",
    tags=["mlops", "training", "abalone", "production"]
)

if __name__ == "__main__":
    # Apply the deployment
    deployment_id = deployment.apply()
    print(f"🚀 Deployment created with ID: {deployment_id}")
    print("📅 Scheduled to run daily at 2 AM UTC")
    print("🔗 View in Prefect UI: http://localhost:4200")
