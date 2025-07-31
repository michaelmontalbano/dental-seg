from sagemaker.pytorch import PyTorch
import sagemaker
import boto3

# Get the SageMaker role and session
role = 'arn:aws:iam::552401576656:role/SageMakerExecutionRole'
session = sagemaker.Session()

# Define the training job with Vision Transformer
estimator = PyTorch(
    entry_point='train_vit.py',
    source_dir='.',  # includes train_vit.py
    role=role,
    framework_version='2.1.0',
    py_version='py310',
    instance_type='ml.g6.8xlarge',
    instance_count=1,
    hyperparameters={
        'data_dir': '/opt/ml/input/data/classification',
        'model_size': 'base',  # tiny, small, base, large
        'epochs': 150,
        'batch_size': 32,
        'learning_rate': 0.0001,
    },
    # Removed dependencies - installing in script instead
    output_path=f's3://{session.default_bucket()}/vit-training-output',
    base_job_name='vit-company-classification'
)

# Point to S3 dataset root with ImageFolder structure: train/[company_name]/*.jpg, val/[company_name]/*.jpg
estimator.fit({'classification': 's3://codentist-general/datasets/aug_dataset/'})
