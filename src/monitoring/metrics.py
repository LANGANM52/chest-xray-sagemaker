"""
Monitoring and metrics collection for the ML system.
Integrates with Prometheus for observability.
"""
import time
import logging
from typing import Optional
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from functools import wraps
import boto3
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define Prometheus metrics
PREDICTION_COUNTER = Counter(
    'model_predictions_total',
    'Total number of predictions',
    ['endpoint_name', 'status']
)

PREDICTION_LATENCY = Histogram(
    'model_prediction_latency_seconds',
    'Prediction latency in seconds',
    ['endpoint_name'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

PREDICTION_CONFIDENCE = Histogram(
    'model_prediction_confidence',
    'Prediction confidence scores',
    ['endpoint_name', 'disease'],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

MODEL_ACCURACY = Gauge(
    'model_accuracy',
    'Model accuracy from validation',
    ['endpoint_name']
)

ENDPOINT_INVOCATIONS = Counter(
    'sagemaker_endpoint_invocations_total',
    'Total SageMaker endpoint invocations',
    ['endpoint_name']
)

ENDPOINT_ERRORS = Counter(
    'sagemaker_endpoint_errors_total',
    'Total SageMaker endpoint errors',
    ['endpoint_name', 'error_type']
)


class MetricsCollector:
    """Collects and exposes metrics for monitoring"""
    
    def __init__(self, port: int = 9090):
        self.port = port
        self.endpoint_name = None
        
    def start_server(self):
        """Start Prometheus metrics server"""
        logger.info(f"Starting Prometheus metrics server on port {self.port}")
        start_http_server(self.port)
        logger.info(f"Metrics available at http://localhost:{self.port}/metrics")
    
    def set_endpoint_name(self, endpoint_name: str):
        """Set the endpoint name for metrics"""
        self.endpoint_name = endpoint_name
    
    def record_prediction(
        self,
        latency: float,
        predictions: list,
        status: str = 'success'
    ):
        """Record a prediction event"""
        
        if not self.endpoint_name:
            logger.warning("Endpoint name not set, skipping metrics")
            return
        
        # Increment prediction counter
        PREDICTION_COUNTER.labels(
            endpoint_name=self.endpoint_name,
            status=status
        ).inc()
        
        # Record latency
        PREDICTION_LATENCY.labels(
            endpoint_name=self.endpoint_name
        ).observe(latency)
        
        # Record confidence scores
        for pred in predictions:
            PREDICTION_CONFIDENCE.labels(
                endpoint_name=self.endpoint_name,
                disease=pred['disease']
            ).observe(pred['probability'])
    
    def record_error(self, error_type: str = 'unknown'):
        """Record an error"""
        
        if not self.endpoint_name:
            return
        
        ENDPOINT_ERRORS.labels(
            endpoint_name=self.endpoint_name,
            error_type=error_type
        ).inc()


class CloudWatchMetrics:
    """Integration with AWS CloudWatch metrics"""
    
    def __init__(self, region: str = 'us-east-1'):
        self.cloudwatch = boto3.client('cloudwatch', region_name=region)
        self.namespace = 'ChestXRayML'
    
    def put_custom_metric(
        self,
        metric_name: str,
        value: float,
        unit: str = 'None',
        dimensions: dict = None
    ):
        """Put custom metric to CloudWatch"""
        
        metric_data = {
            'MetricName': metric_name,
            'Value': value,
            'Unit': unit,
            'Timestamp': datetime.utcnow()
        }
        
        if dimensions:
            metric_data['Dimensions'] = [
                {'Name': k, 'Value': v} for k, v in dimensions.items()
            ]
        
        try:
            self.cloudwatch.put_metric_data(
                Namespace=self.namespace,
                MetricData=[metric_data]
            )
        except Exception as e:
            logger.error(f"Error putting metric to CloudWatch: {str(e)}")
    
    def get_endpoint_metrics(
        self,
        endpoint_name: str,
        metric_name: str = 'Invocations',
        hours: int = 24
    ):
        """Get SageMaker endpoint metrics from CloudWatch"""
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        response = self.cloudwatch.get_metric_statistics(
            Namespace='AWS/SageMaker',
            MetricName=metric_name,
            Dimensions=[
                {
                    'Name': 'EndpointName',
                    'Value': endpoint_name
                }
            ],
            StartTime=start_time,
            EndTime=end_time,
            Period=3600,  # 1 hour
            Statistics=['Sum', 'Average']
        )
        
        return response['Datapoints']
    
    def log_prediction_metrics(
        self,
        endpoint_name: str,
        latency: float,
        predictions: list
    ):
        """Log prediction metrics to CloudWatch"""
        
        # Latency
        self.put_custom_metric(
            metric_name='PredictionLatency',
            value=latency,
            unit='Milliseconds',
            dimensions={'EndpointName': endpoint_name}
        )
        
        # Top prediction confidence
        if predictions:
            top_confidence = predictions[0]['probability']
            self.put_custom_metric(
                metric_name='TopPredictionConfidence',
                value=top_confidence,
                dimensions={'EndpointName': endpoint_name}
            )


def monitor_prediction(collector: MetricsCollector):
    """Decorator to monitor predictions"""
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status = 'success'
            predictions = None
            
            try:
                result = func(*args, **kwargs)
                predictions = result.get('predictions', [])
                return result
            except Exception as e:
                status = 'error'
                collector.record_error(error_type=type(e).__name__)
                raise
            finally:
                latency = time.time() - start_time
                
                if predictions:
                    collector.record_prediction(
                        latency=latency,
                        predictions=predictions,
                        status=status
                    )
        
        return wrapper
    return decorator


class ModelMonitor:
    """Monitor model performance and data drift"""
    
    def __init__(self, endpoint_name: str, region: str = 'us-east-1'):
        self.endpoint_name = endpoint_name
        self.sagemaker = boto3.client('sagemaker', region_name=region)
    
    def create_monitoring_schedule(
        self,
        baseline_dataset_uri: str,
        output_uri: str,
        schedule_expression: str = 'cron(0 0 * * ? *)'  # Daily at midnight
    ):
        """Create SageMaker Model Monitor schedule"""
        
        monitoring_schedule_name = f"{self.endpoint_name}-monitor"
        
        try:
            self.sagemaker.create_model_quality_job_definition(
                JobDefinitionName=f"{monitoring_schedule_name}-job-def",
                ModelQualityAppSpecification={
                    'ImageUri': 'your-monitoring-image-uri',  # Replace with actual
                    'ProblemType': 'MulticlassClassification'
                },
                ModelQualityJobInput={
                    'EndpointInput': {
                        'EndpointName': self.endpoint_name,
                        'LocalPath': '/opt/ml/processing/input'
                    }
                },
                ModelQualityJobOutputConfig={
                    'MonitoringOutputs': [
                        {
                            'S3Output': {
                                'S3Uri': output_uri,
                                'LocalPath': '/opt/ml/processing/output'
                            }
                        }
                    ]
                },
                RoleArn='your-role-arn'  # Replace with actual
            )
            
            logger.info(f"✓ Model monitoring schedule created: {monitoring_schedule_name}")
            
        except Exception as e:
            logger.error(f"Error creating monitoring schedule: {str(e)}")


# Global metrics collector instance
metrics_collector = MetricsCollector()


if __name__ == '__main__':
    # Start metrics server for testing
    metrics_collector.start_server()
    
    logger.info("Metrics server running. Press Ctrl+C to stop.")
    
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down metrics server")
