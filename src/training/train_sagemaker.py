"""
SageMaker training job orchestration script.
Launches training jobs with experiment tracking and cost optimization.
"""
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.experiments import Run
from sagemaker.inputs import TrainingInput
import click
import logging
from datetime import datetime
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SageMakerTrainer:
    """Orchestrates SageMaker training jobs"""
    
    def __init__(
        self,
        role: str = None,
        region: str = 'us-east-1',
        bucket: str = None,
        experiment_name: str = 'chest-xray-classification'
    ):
        self.session = sagemaker.Session()
        self.region = region
        self.bucket = bucket or self.session.default_bucket()
        self.experiment_name = experiment_name
        
        # Get execution role
        if role:
            self.role = role
        else:
            self.role = sagemaker.get_execution_role()
        
        logger.info(f"Using role: {self.role}")
        logger.info(f"Using bucket: {self.bucket}")
    
    def launch_training_job(
        self,
        instance_type: str = 'ml.g4dn.xlarge',
        instance_count: int = 1,
        use_spot: bool = True,
        max_run: int = 43200,
        max_wait: int = 86400,
        volume_size: int = 50,
        hyperparameters: dict = None,
        trial_name: str = None
    ):
        """Launch a SageMaker training job"""
        
        # Default hyperparameters
        if hyperparameters is None:
            hyperparameters = {
                'num-epochs': 10,
                'batch-size': 32,
                'learning-rate': 0.001,
                'image-size': 224,
                'num-classes': 14
            }
        
        # Training input channels
        train_input = TrainingInput(
            s3_data=f's3://{self.bucket}/data/splits/train.csv',
            content_type='text/csv'
        )
        
        val_input = TrainingInput(
            s3_data=f's3://{self.bucket}/data/splits/val.csv',
            content_type='text/csv'
        )
        
        # Checkpoint configuration
        checkpoint_s3_uri = f's3://{self.bucket}/checkpoints'
        
        # Generate job name
        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        job_name = f'chest-xray-{timestamp}'
        
        # Configure PyTorch estimator
        estimator = PyTorch(
            entry_point='train.py',
            source_dir='./src/training',
            role=self.role,
            instance_type=instance_type,
            instance_count=instance_count,
            framework_version='2.0',
            py_version='py310',
            hyperparameters=hyperparameters,
            output_path=f's3://{self.bucket}/output',
            code_location=f's3://{self.bucket}/code',
            checkpoint_s3_uri=checkpoint_s3_uri,
            checkpoint_local_path='/opt/ml/checkpoints',
            use_spot_instances=use_spot,
            max_run=max_run,
            max_wait=max_wait if use_spot else None,
            volume_size=volume_size,
            keep_alive_period_in_seconds=600 if use_spot else 0,
            base_job_name='chest-xray',
            enable_sagemaker_metrics=True,
            metric_definitions=[
                {'Name': 'train:loss', 'Regex': 'Train Loss: ([0-9\\.]+)'},
                {'Name': 'validation:loss', 'Regex': 'Val Loss: ([0-9\\.]+)'},
                {'Name': 'validation:auc', 'Regex': 'Val AUC: ([0-9\\.]+)'}
            ]
        )
        
        # Launch with experiment tracking
        if trial_name is None:
            trial_name = job_name
        
        logger.info(f"Launching training job: {job_name}")
        logger.info(f"Instance: {instance_type} (Spot: {use_spot})")
        logger.info(f"Experiment: {self.experiment_name}, Trial: {trial_name}")
        
        with Run(
            experiment_name=self.experiment_name,
            run_name=trial_name,
            sagemaker_session=self.session
        ) as run:
            # Log parameters
            run.log_parameters({
                'instance_type': instance_type,
                'use_spot': use_spot,
                **hyperparameters
            })
            
            # Fit the model
            estimator.fit(
                inputs={
                    'train': train_input,
                    'validation': val_input
                },
                job_name=job_name,
                wait=False  # Don't block, return immediately
            )
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Training job launched: {job_name}")
        logger.info(f"{'='*60}")
        logger.info(f"Monitor progress:")
        logger.info(f"  AWS Console: https://console.aws.amazon.com/sagemaker/home?region={self.region}#/jobs/{job_name}")
        logger.info(f"  CloudWatch Logs: https://console.aws.amazon.com/cloudwatch/home?region={self.region}#logStream:group=/aws/sagemaker/TrainingJobs;prefix={job_name}")
        logger.info(f"\nEstimated cost (spot): ~$2-5 for 10 epochs")
        logger.info(f"Estimated time: 30-60 minutes")
        logger.info(f"\nTo wait for completion, run:")
        logger.info(f"  estimator.latest_training_job.wait()")
        
        return estimator, job_name
    
    def get_training_job_status(self, job_name: str):
        """Get the status of a training job"""
        
        sm_client = boto3.client('sagemaker', region_name=self.region)
        response = sm_client.describe_training_job(TrainingJobName=job_name)
        
        status = response['TrainingJobStatus']
        logger.info(f"Job {job_name}: {status}")
        
        if status == 'Completed':
            model_artifacts = response['ModelArtifacts']['S3ModelArtifacts']
            logger.info(f"Model artifacts: {model_artifacts}")
        elif status == 'Failed':
            logger.error(f"Failure reason: {response.get('FailureReason', 'Unknown')}")
        
        return response


@click.command()
@click.option('--instance-type', default='ml.g4dn.xlarge', help='Training instance type')
@click.option('--use-spot/--no-spot', default=True, help='Use spot instances')
@click.option('--num-epochs', default=10, help='Number of training epochs')
@click.option('--batch-size', default=32, help='Batch size')
@click.option('--learning-rate', default=0.001, help='Learning rate')
@click.option('--bucket', default=None, help='S3 bucket name')
@click.option('--role', default=None, help='SageMaker execution role ARN')
@click.option('--wait/--no-wait', default=False, help='Wait for job to complete')
def main(instance_type, use_spot, num_epochs, batch_size, learning_rate, bucket, role, wait):
    """Launch SageMaker training job for chest X-ray classification"""
    
    try:
        # Initialize trainer
        trainer = SageMakerTrainer(role=role, bucket=bucket)
        
        # Hyperparameters
        hyperparameters = {
            'num-epochs': num_epochs,
            'batch-size': batch_size,
            'learning-rate': learning_rate,
            'image-size': 224,
            'num-classes': 14
        }
        
        # Launch training
        estimator, job_name = trainer.launch_training_job(
            instance_type=instance_type,
            use_spot=use_spot,
            hyperparameters=hyperparameters
        )
        
        if wait:
            logger.info("\nWaiting for training job to complete...")
            estimator.latest_training_job.wait()
            
            # Get final status
            trainer.get_training_job_status(job_name)
        
        logger.info("\n✓ Training job launched successfully!")
        
    except Exception as e:
        logger.error(f"Error launching training job: {str(e)}")
        raise


if __name__ == '__main__':
    main()
