"""
Deploy trained model to SageMaker Serverless Inference endpoint.
Cost-optimized deployment with auto-scaling.
"""
import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker.serverless import ServerlessInferenceConfig
import click
import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelDeployer:
    """Handles model deployment to SageMaker endpoints"""
    
    def __init__(
        self,
        role: str = None,
        region: str = 'us-east-1',
        bucket: str = None
    ):
        self.session = sagemaker.Session()
        self.region = region
        self.bucket = bucket or self.session.default_bucket()
        
        # Get execution role
        if role:
            self.role = role
        else:
            self.role = sagemaker.get_execution_role()
        
        logger.info(f"Using role: {self.role}")
        logger.info(f"Using bucket: {self.bucket}")
    
    def deploy_serverless(
        self,
        model_data: str,
        endpoint_name: str = None,
        memory_size: int = 4096,
        max_concurrency: int = 10
    ):
        """
        Deploy model to serverless endpoint.
        
        Args:
            model_data: S3 URI to model.tar.gz
            endpoint_name: Name for the endpoint
            memory_size: Memory in MB (2048, 4096, or 6144)
            max_concurrency: Max concurrent invocations (1-200)
        """
        
        if endpoint_name is None:
            timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M')
            endpoint_name = f'chest-xray-serverless-{timestamp}'
        
        logger.info(f"Deploying serverless endpoint: {endpoint_name}")
        logger.info(f"Model data: {model_data}")
        
        # Create PyTorch model
        model = PyTorchModel(
            model_data=model_data,
            role=self.role,
            framework_version='2.0',
            py_version='py310',
            entry_point='inference.py',
            source_dir='./src/inference',
            sagemaker_session=self.session
        )
        
        # Serverless configuration
        serverless_config = ServerlessInferenceConfig(
            memory_size_in_mb=memory_size,
            max_concurrency=max_concurrency
        )
        
        logger.info("Deploying model... (this may take 5-10 minutes)")
        
        # Deploy
        predictor = model.deploy(
            serverless_inference_config=serverless_config,
            endpoint_name=endpoint_name
        )
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✓ Endpoint deployed successfully!")
        logger.info(f"{'='*60}")
        logger.info(f"Endpoint name: {endpoint_name}")
        logger.info(f"Memory: {memory_size} MB")
        logger.info(f"Max concurrency: {max_concurrency}")
        logger.info(f"\nCost structure:")
        logger.info(f"  - No idle costs (pay only for inference time)")
        logger.info(f"  - ~$0.20 per 1M inference milliseconds")
        logger.info(f"  - Example: 1000 predictions (~1s each) = ~$0.20")
        logger.info(f"\nTest your endpoint:")
        logger.info(f"  python src/inference/test_endpoint.py --endpoint-name {endpoint_name}")
        
        return predictor, endpoint_name
    
    def deploy_real_time(
        self,
        model_data: str,
        endpoint_name: str = None,
        instance_type: str = 'ml.t2.medium',
        instance_count: int = 1
    ):
        """
        Deploy model to real-time endpoint (for comparison/testing).
        Note: Real-time endpoints have idle costs!
        """
        
        if endpoint_name is None:
            timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M')
            endpoint_name = f'chest-xray-realtime-{timestamp}'
        
        logger.info(f"Deploying real-time endpoint: {endpoint_name}")
        
        # Create PyTorch model
        model = PyTorchModel(
            model_data=model_data,
            role=self.role,
            framework_version='2.0',
            py_version='py310',
            entry_point='inference.py',
            source_dir='./src/inference',
            sagemaker_session=self.session
        )
        
        logger.info("Deploying model... (this may take 5-10 minutes)")
        
        # Deploy
        predictor = model.deploy(
            instance_type=instance_type,
            initial_instance_count=instance_count,
            endpoint_name=endpoint_name
        )
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✓ Endpoint deployed successfully!")
        logger.info(f"{'='*60}")
        logger.info(f"Endpoint name: {endpoint_name}")
        logger.info(f"Instance: {instance_type}")
        logger.info(f"\n⚠️  Warning: This endpoint has IDLE COSTS!")
        logger.info(f"  Hourly cost: ~$0.05-0.10 (ml.t2.medium)")
        logger.info(f"  Remember to delete when done testing!")
        
        return predictor, endpoint_name
    
    def delete_endpoint(self, endpoint_name: str):
        """Delete an endpoint to stop costs"""
        
        logger.info(f"Deleting endpoint: {endpoint_name}")
        
        sm_client = boto3.client('sagemaker', region_name=self.region)
        
        try:
            # Delete endpoint
            sm_client.delete_endpoint(EndpointName=endpoint_name)
            logger.info("✓ Endpoint deleted")
            
            # Delete endpoint configuration
            sm_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
            logger.info("✓ Endpoint configuration deleted")
            
            logger.info("\nEndpoint cleanup complete!")
            
        except Exception as e:
            logger.error(f"Error deleting endpoint: {str(e)}")
    
    def list_endpoints(self):
        """List all active endpoints"""
        
        sm_client = boto3.client('sagemaker', region_name=self.region)
        
        response = sm_client.list_endpoints(
            SortBy='CreationTime',
            SortOrder='Descending',
            MaxResults=50
        )
        
        endpoints = response.get('Endpoints', [])
        
        if not endpoints:
            logger.info("No active endpoints found")
            return
        
        logger.info(f"\nActive endpoints ({len(endpoints)}):")
        logger.info("-" * 80)
        
        for ep in endpoints:
            logger.info(f"Name: {ep['EndpointName']}")
            logger.info(f"  Status: {ep['EndpointStatus']}")
            logger.info(f"  Created: {ep['CreationTime']}")
            logger.info("-" * 80)
    
    def get_latest_model(self):
        """Get the latest trained model from S3"""
        
        s3_client = boto3.client('s3')
        
        # List objects in output folder
        prefix = 'output/'
        response = s3_client.list_objects_v2(
            Bucket=self.bucket,
            Prefix=prefix
        )
        
        if 'Contents' not in response:
            logger.error("No trained models found in S3")
            return None
        
        # Find most recent model.tar.gz
        model_objects = [
            obj for obj in response['Contents']
            if obj['Key'].endswith('model.tar.gz')
        ]
        
        if not model_objects:
            logger.error("No model.tar.gz files found")
            return None
        
        # Sort by last modified
        latest = sorted(model_objects, key=lambda x: x['LastModified'], reverse=True)[0]
        model_uri = f"s3://{self.bucket}/{latest['Key']}"
        
        logger.info(f"Latest model: {model_uri}")
        return model_uri


@click.command()
@click.option('--model-data', default=None, help='S3 URI to model.tar.gz')
@click.option('--endpoint-name', default=None, help='Endpoint name')
@click.option('--deployment-type', type=click.Choice(['serverless', 'realtime']), 
              default='serverless', help='Deployment type')
@click.option('--memory-size', type=int, default=4096, 
              help='Memory size in MB (serverless only)')
@click.option('--max-concurrency', type=int, default=10,
              help='Max concurrent requests (serverless only)')
@click.option('--instance-type', default='ml.t2.medium',
              help='Instance type (realtime only)')
@click.option('--bucket', default=None, help='S3 bucket name')
@click.option('--role', default=None, help='SageMaker execution role ARN')
@click.option('--list-endpoints', is_flag=True, help='List all endpoints')
@click.option('--delete-endpoint', default=None, help='Delete an endpoint')
def main(
    model_data, endpoint_name, deployment_type, memory_size, 
    max_concurrency, instance_type, bucket, role, 
    list_endpoints, delete_endpoint
):
    """Deploy model to SageMaker inference endpoint"""
    
    try:
        deployer = ModelDeployer(role=role, bucket=bucket)
        
        # List endpoints
        if list_endpoints:
            deployer.list_endpoints()
            return
        
        # Delete endpoint
        if delete_endpoint:
            deployer.delete_endpoint(delete_endpoint)
            return
        
        # Get model data
        if model_data is None:
            logger.info("No model data specified, looking for latest model...")
            model_data = deployer.get_latest_model()
            
            if model_data is None:
                logger.error("Please specify --model-data or train a model first")
                return
        
        # Deploy
        if deployment_type == 'serverless':
            predictor, ep_name = deployer.deploy_serverless(
                model_data=model_data,
                endpoint_name=endpoint_name,
                memory_size=memory_size,
                max_concurrency=max_concurrency
            )
        else:
            predictor, ep_name = deployer.deploy_real_time(
                model_data=model_data,
                endpoint_name=endpoint_name,
                instance_type=instance_type
            )
        
        logger.info("\n✓ Deployment complete!")
        
    except Exception as e:
        logger.error(f"Error during deployment: {str(e)}")
        raise


if __name__ == '__main__':
    main()
