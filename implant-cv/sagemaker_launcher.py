from sagemaker.pytorch import PyTorch
import sagemaker
import boto3

# Get the SageMaker role and session
role = 'arn:aws:iam::552401576656:role/SageMakerExecutionRole'
session = sagemaker.Session()

# Define the training job
estimator = PyTorch(
    entry_point='train.py',
    source_dir='.',  # assumes train.py and data.yaml are in the same folder
    role=role,
    framework_version='2.1.0',
    py_version='py310',
    instance_type='ml.g6.8xlarge',  # Change as needed
    instance_count=1,
    hyperparameters={
       # 'data_yaml': '/opt/ml/input/data/train/data.yaml',
        'data_yaml': '/opt/ml/code/data.yaml',  # Path to data.yaml in the source_dir
        'model': 'yolov8m.yaml',
        'epochs': 1,
        'batch': 16,
        'imgsz': 256,
        'project': 'runs/train',
        'name': 'implants-v1',
        'learning_rate': 0.01,
        'workers': 4,
    },
    dependencies=['requirements.txt'],
    output_path=f's3://{session.default_bucket()}/yolo-training-output',
    base_job_name='yolov8-implants'
)

# Point to S3 dataset root; this is mounted to /opt/ml/input/data/train/
estimator.fit({'train': 's3://codentist-general/datasets/merged_implants/'})
