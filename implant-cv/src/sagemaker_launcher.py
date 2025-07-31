import sagemaker
from sagemaker.pytorch import PyTorch
import boto3
import os

role = sagemaker.get_execution_role()
region = boto3.Session().region_name
session = sagemaker.Session()

# Training input
input_data_uri = "s3://codentist-general/datasets/implants/export_169465_project-169465-at-2025-07-24-03-20-1505fd09.json"
image_data_uri = "s3://codentist-general/datasets/implants/images/"

# Upload both to a training channel
input_channel = session.upload_data(path='.', key_prefix='input/json')  # optionally upload locally for testing

estimator = PyTorch(
    entry_point='train.py',
    source_dir='.',  # directory with train.py, dataset.py, requirements.txt
    role=role,
    framework_version='2.1.0',
    py_version='py310',
    instance_count=1,
    instance_type='ml.g4dn.xlarge',
    hyperparameters={
        "json_path": input_data_uri,
        "image_root": image_data_uri,
        "num_classes": 15,
        "epochs": 50,
        "batch_size": 32
    },
    dependencies=['requirements.txt'],
    output_path=f"s3://{session.default_bucket()}/training-output",
    base_job_name='implant-classifier'
)

# You can choose either of these depending on where you host the JSON
estimator.fit()

