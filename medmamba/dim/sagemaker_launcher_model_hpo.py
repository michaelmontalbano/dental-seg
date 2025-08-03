import os
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.tuner import HyperparameterTuner, IntegerParameter, ContinuousParameter, CategoricalParameter
from datetime import datetime

# Get SageMaker session and role
session = sagemaker.Session()
role = sagemaker.get_execution_role()

# S3 paths
bucket = session.default_bucket()
base_job_name = f"model-classifier-hpo-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
output_path = f"s3://{bucket}/sagemaker/model-classifier-hpo"

# Define the estimator with fixed parameters
estimator = PyTorch(
    entry_point='train_model_classifier_hpo.py',
    source_dir='.',
    role=role,
    instance_type='ml.g4dn.xlarge',
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
    }
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
    estimator,
    objective_metric_name,
    hyperparameter_ranges,
    max_jobs=20,
    max_parallel_jobs=4,
    objective_type=objective_type,
    base_tuning_job_name='model-class-hpo',
    strategy='Bayesian'
)

# Data configuration
data_channels = {
    'train_augmented': f"s3://{bucket}/sagemaker/implants/augmented-data/"
}

print(f"\n🚀 Starting hyperparameter tuning job...")
print(f"📊 Optimizing for: {objective_metric_name} ({objective_type})")
print(f"🔍 Total jobs: 20 (4 parallel)")
print(f"📈 Strategy: Bayesian optimization")
print(f"\n💾 Results will be saved to: {output_path}")

# Start the tuning job
tuner.fit(data_channels)

print(f"\n✅ Tuning job submitted!")
print(f"📊 Monitor progress in SageMaker console")
print(f"⏰ Estimated time: 3-5 hours")

# Print out how to analyze results
print(f"\n📈 To analyze results after completion:")
print(f"cd ../company_embeddings")
print(f"python analyze_tuning_results.py --tuning-job-name {tuner.latest_tuning_job.name}")
