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
base_job_name = f"dual-embeddings-hpo-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
output_path = f"s3://{bucket}/sagemaker/dual-embeddings-hpo"

# Checkpoint paths
company_checkpoint_path = f"s3://codentist-general/models/implant_company/vit_company.pth"
model_checkpoint_path = f"s3://codentist-general/models/implant_model_type/best_model.pth"

# Define the estimator with fixed parameters
estimator = PyTorch(
    entry_point='train_dual_embeddings_regression.py',
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
        'task': 'both',
        'validate_every': 1,
        'save_best_only': 'true',
        'use_attention': 'true',
        'company_checkpoint': '/opt/ml/input/data/company_checkpoint/vit_company.pth',
        'model_checkpoint': '/opt/ml/input/data/model_checkpoint/best_model.pth',
        'epochs': 100,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'warmup_epochs': 5,
        'company_lr_multiplier': 0.1,
        'model_lr_multiplier': 0.1
    },
    metric_definitions=[
        {
            'Name': 'avg_mae',
            'Regex': '\\[(\\d+)\\] avg_mae = ([0-9\\.]+);'
        },
        {
            'Name': 'val_loss',
            'Regex': '\\[(\\d+)\\] val_loss = ([0-9\\.]+);'
        },
        {
            'Name': 'length_mae',
            'Regex': '\\[(\\d+)\\] length_mae = ([0-9\\.]+);'
        },
        {
            'Name': 'diameter_mae', 
            'Regex': '\\[(\\d+)\\] diameter_mae = ([0-9\\.]+);'
        }
    ]
)

# Define hyperparameter ranges
hyperparameter_ranges = {
    'model_size': CategoricalParameter(['tiny', 'small', 'base']),
    'weight_decay': ContinuousParameter(0.01, 0.1, scaling_type='Logarithmic'),
    'company_embed_dim': CategoricalParameter([128, 256, 512]),
    'model_embed_dim': CategoricalParameter([128, 256, 512]),
    'attention_type': CategoricalParameter(['dual', 'triple', 'hierarchical']),
    'unfreeze_epoch': IntegerParameter(15, 30),
    'full_unfreeze_epoch': IntegerParameter(30, 60)
}

metric_definitions=[
    {
        'Name': 'avg_mae',
        'Regex': r'\[(\d+)\] avg_mae = ([0-9\.]+);'
    },
    {
        'Name': 'val_loss',
        'Regex': r'\[(\d+)\] val_loss = ([0-9\.]+);'
    },
    {
        'Name': 'length_mae',
        'Regex': r'\[(\d+)\] length_mae = ([0-9\.]+);'
    },
    {
        'Name': 'diameter_mae', 
        'Regex': r'\[(\d+)\] diameter_mae = ([0-9\.]+);'
    }
]


# Metric to optimize
objective_metric_name = 'avg_mae'
objective_type = 'Minimize'


# Create the tuner
tuner = HyperparameterTuner(
    estimator=estimator,
    objective_metric_name='avg_mae',
    objective_type='Minimize',
    hyperparameter_ranges=hyperparameter_ranges,
    metric_definitions=[
        {'Name': 'avg_mae', 'Regex': 'avg_mae: ([0-9\\.]+)'}
    ],
    max_jobs=16,
    max_parallel_jobs=4,
    base_tuning_job_name='dual-embeddings-hpo'
)

# Data configuration
data_channels = {
    'train_augmented': "s3://codentist-general/datasets/aug_dataset",
    'company_checkpoint': company_checkpoint_path,
    'model_checkpoint': model_checkpoint_path
}

print(f"\n🚀 Starting dual embeddings hyperparameter tuning job...")
print(f"📊 Optimizing for: {objective_metric_name} ({objective_type})")
print(f"🔍 Total jobs: 8 (4 parallel)")
print(f"📈 Strategy: Bayesian optimization")
print(f"\n💾 Results will be saved to: {output_path}")
print(f"🏢 Company checkpoint: {company_checkpoint_path}")
print(f"🔧 Model checkpoint: {model_checkpoint_path}")

print(f"\n📋 Hyperparameter space (7 parameters):")
print(f"  - model_size: [tiny, small, base]")
print(f"  - weight_decay: 0.01-0.1")
print(f"  - company_embed_dim: [128, 256, 512]")
print(f"  - model_embed_dim: [128, 256, 512]")
print(f"  - attention_type: [dual, triple, hierarchical]")
print(f"  - unfreeze_epoch: 15-30")
print(f"  - full_unfreeze_epoch: 30-60")

# Start the tuning job
tuner.fit(data_channels)

print(f"\n✅ Tuning job submitted!")
print(f"📊 Monitor progress in SageMaker console")
print(f"⏰ Estimated time: 3-4 hours (8 jobs @ 100 epochs each)")

# Print out how to analyze results
print(f"\n📈 To analyze results after completion:")
print(f"python analyze_tuning_results.py --tuning-job-name {tuner.latest_tuning_job.name}")
