import os
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.tuner import HyperparameterTuner, IntegerParameter, ContinuousParameter, CategoricalParameter
from datetime import datetime

# Get SageMaker session and role
session = sagemaker.Session()
role = 'arn:aws:iam::552401576656:role/SageMakerExecutionRole'

# S3 paths
bucket = session.default_bucket()
base_job_name = f"company-classifier-hpo-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
output_path = f"s3://{bucket}/sagemaker/company-classifier-hpo"

# Define metric definitions separately
metric_definitions = [
    {
        'Name': 'train_loss',
        'Regex': r'\[(\d+)\] train_loss = ([0-9\.]+);'
    },
    {
        'Name': 'train_accuracy',
        'Regex': r'\[(\d+)\] train_accuracy = ([0-9\.]+);'
    },
    {
        'Name': 'val_loss',
        'Regex': r'\[(\d+)\] val_loss = ([0-9\.]+);'
    },
    {
        'Name': 'val_accuracy',
        'Regex': r'\[(\d+)\] val_accuracy = ([0-9\.]+);'
    },
    {
        'Name': 'best_val_accuracy',
        'Regex': r'best_val_accuracy = ([0-9\.]+);'
    }
]

# Define the estimator with fixed parameters
estimator = PyTorch(
    entry_point='train_company_classifier_hpo.py',
    source_dir='.',
    role=role,
    instance_type='ml.g6.8xlarge',
    instance_count=1,
    framework_version='2.0.0',
    py_version='py310',
    base_job_name=base_job_name,
    output_path=output_path,
    sagemaker_session=session,
    environment={
        'WANDB_DISABLED': 'true',
        'TOKENIZERS_PARALLELISM': 'false'
    },
    hyperparameters={
        'train_json': '/opt/ml/input/data/train_augmented/augmented_train.json',
        'val_json': '/opt/ml/input/data/train_augmented/augmented_val.json',
        'image_root': '/opt/ml/input/data/train_augmented',
        'output_dir': '/opt/ml/model',
        'validate_every': 1,
        'save_best_only': 'true'
    },
    metric_definitions=metric_definitions
)

# Define hyperparameter ranges
hyperparameter_ranges = {
    'model_size': CategoricalParameter(['tiny', 'small', 'base']),
    'epochs': IntegerParameter(50, 100),
    'batch_size': CategoricalParameter([16, 32, 64]),
    'learning_rate': ContinuousParameter(5e-5, 5e-4, scaling_type='Logarithmic'),
    'weight_decay': ContinuousParameter(0.01, 0.1, scaling_type='Logarithmic'),
    'warmup_epochs': IntegerParameter(3, 10)
}

# Metric to optimize
objective_metric_name = 'val_accuracy'
objective_type = 'Maximize'

# Create the tuner
tuner = HyperparameterTuner(
    estimator=estimator,
    objective_metric_name=objective_metric_name,
    objective_type=objective_type,
    hyperparameter_ranges=hyperparameter_ranges,
    metric_definitions=[
        {'Name': 'val_accuracy', 'Regex': r'\[\d+\] val_accuracy = ([0-9\.]+);'}
    ],
    max_jobs=20,
    max_parallel_jobs=4,
    base_tuning_job_name='company-class-hpo',
    strategy='Bayesian'
)

# Data configuration
data_channels = {
    'train_augmented': "s3://codentist-general/datasets/aug_dataset"
}

print(f"\n🚀 Starting company classifier hyperparameter tuning job...")
print(f"📊 Optimizing for: {objective_metric_name} ({objective_type})")
print(f"🔍 Total jobs: 20 (4 parallel)")
print(f"📈 Strategy: Bayesian optimization")
print(f"\n💾 Results will be saved to: {output_path}")

print(f"\n📋 Hyperparameter space (6 parameters):")
print(f"  - model_size: [tiny, small, base]")
print(f"  - epochs: 50-100")
print(f"  - batch_size: [16, 32, 64]")
print(f"  - learning_rate: 5e-5 to 5e-4")
print(f"  - weight_decay: 0.01-0.1")
print(f"  - warmup_epochs: 3-10")

# Start the tuning job
tuner.fit(data_channels)

print(f"\n✅ Tuning job submitted!")
print(f"📊 Monitor progress in SageMaker console")
print(f"⏰ Estimated time: 4-6 hours (20 jobs @ ~75 epochs avg)")

# Print out how to analyze results
print(f"\n📈 To analyze results after completion:")
print(f"python analyze_tuning_results.py --tuning-job-name {tuner.latest_tuning_job.name}")
