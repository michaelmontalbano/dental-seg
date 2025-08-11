from sagemaker.pytorch import PyTorch
from sagemaker.tuner import HyperparameterTuner, IntegerParameter, ContinuousParameter, CategoricalParameter
import sagemaker
import argparse
import time
import json

def create_hyperparameter_tuner(task='diameter', max_jobs=20, max_parallel_jobs=5):
    """Create hyperparameter tuner for regression tasks"""
    
    role = 'arn:aws:iam::552401576656:role/SageMakerExecutionRole'
    session = sagemaker.Session()
    bucket = 'codentist-general'
    
    # Define hyperparameter ranges
    hyperparameter_ranges = {
        'learning_rate': ContinuousParameter(1e-5, 1e-3, scaling_type='Logarithmic'),
        'batch_size': CategoricalParameter([16, 32, 64]),
        'model_size': CategoricalParameter(['tiny', 'small', 'base']),
        'embed_dim': CategoricalParameter([128, 256, 512]),
        'unfreeze_epoch': IntegerParameter(10, 30),
        'full_unfreeze_epoch': IntegerParameter(30, 50),
        'company_lr_multiplier': ContinuousParameter(0.01, 0.5, scaling_type='Logarithmic'),
        'weight_decay': ContinuousParameter(0.01, 0.1, scaling_type='Logarithmic'),
        'use_attention': CategoricalParameter(['true', 'false']),
    }
    
    # Fixed hyperparameters
    static_hyperparameters = {
        'train_json': '/opt/ml/input/data/train_augmented/augmented_train.json',
        'val_json': '/opt/ml/input/data/train_augmented/augmented_val.json',
        'task': task,  # 'diameter', 'length', or 'both'
        'epochs': 60,  # Reduced for tuning
        'output_dir': '/opt/ml/model',
        'image_root': '/opt/ml/input/data/train_augmented',
        'validate_every': 1,
        'save_best_only': 'true',
        'warmup_epochs': 5,
        'company_checkpoint': 'vit_company.pth'
    }
    
    # Define the estimator
    estimator = PyTorch(
        entry_point='train_regression_with_embeddings.py',
        source_dir='.',
        role=role,
        framework_version='2.1.0',
        py_version='py310',
        instance_type='ml.g6.8xlarge',
        instance_count=1,
        hyperparameters=static_hyperparameters,
        dependencies=['../../requirements.txt'],
        base_job_name=f'hpo-regression-{task}',
        output_path=f's3://{session.default_bucket()}/hpo-regression-{task}-output',
        max_run=8*60*60,  # 8 hours max runtime per job
        keep_alive_period_in_seconds=15*60,  # 15 min keep alive
        enable_sagemaker_metrics=True,  # Enable CloudWatch metrics
    )
    
    # Define objective metric
    if task == 'both':
        # Use average MAE for both tasks
        objective_metric_name = 'avg_mae'
    else:
        # Use task-specific MAE
        objective_metric_name = f'{task}_mae'
    
    objective_type = 'Minimize'
    
    # Create the tuner
    tuner = HyperparameterTuner(
        estimator,
        objective_metric_name,
        hyperparameter_ranges,
        metric_definitions=[
            {'Name': 'val_loss', 'Regex': r'\[\d+\] val_loss = ([\d\.]+);'},
            {'Name': 'length_mae', 'Regex': r'\[\d+\] length_mae = ([\d\.]+);'},
            {'Name': 'diameter_mae', 'Regex': r'\[\d+\] diameter_mae = ([\d\.]+);'},
            {'Name': 'avg_mae', 'Regex': r'\[\d+\] avg_mae = ([\d\.]+);'},
        ],
        max_jobs=max_jobs,
        max_parallel_jobs=max_parallel_jobs,
        objective_type=objective_type,
        strategy='Bayesian',
        early_stopping_type='Auto',  # Enable automatic early stopping
    )
    
    return tuner, estimator

def main():
    parser = argparse.ArgumentParser(description='Launch hyperparameter tuning for regression tasks')
    parser.add_argument('--task', type=str, default='diameter',
                        choices=['length', 'diameter', 'both'],
                        help='Regression task to optimize')
    parser.add_argument('--max_jobs', type=int, default=20,
                        help='Maximum number of training jobs')
    parser.add_argument('--max_parallel_jobs', type=int, default=5,
                        help='Maximum number of parallel training jobs')
    
    args = parser.parse_args()
    
    print(f"🎯 Hyperparameter Tuning for {args.task.upper()} Regression")
    print("=" * 60)
    print(f"📊 Configuration:")
    print(f"   Task: {args.task}")
    print(f"   Max Jobs: {args.max_jobs}")
    print(f"   Max Parallel Jobs: {args.max_parallel_jobs}")
    print(f"   Strategy: Bayesian optimization")
    print(f"   Early Stopping: Enabled")
    print("=" * 60)
    
    # Create tuner
    tuner, estimator = create_hyperparameter_tuner(
        task=args.task,
        max_jobs=args.max_jobs,
        max_parallel_jobs=args.max_parallel_jobs
    )
    
    # Input data configuration
    input_data = {
        'train_augmented': 's3://codentist-general/datasets/aug_dataset/'
    }
    
    print(f"\n📊 Hyperparameter Search Space:")
    print(json.dumps(tuner.hyperparameter_ranges, indent=2, default=str))
    
    print(f"\n📊 Static Hyperparameters:")
    print(json.dumps(estimator.hyperparameters, indent=2))
    
    print(f"\n📊 Objective Metric: {tuner.objective_metric_name} ({tuner.objective_type})")
    
    print(f"\n🚀 Starting hyperparameter tuning job...")
    print(f"   This will launch up to {args.max_jobs} training jobs")
    print(f"   Monitor progress in SageMaker console")
    
    # Launch the tuning job
    tuner.fit(input_data, wait=False)
    
    print(f"\n✅ Hyperparameter tuning job submitted!")
    print(f"📊 Job name: {tuner.latest_tuning_job.name}")
    print(f"📁 Results will be saved to: {estimator.output_path}")
    print(f"\n💡 Tips:")
    print(f"   • Monitor in SageMaker console > Hyperparameter tuning jobs")
    print(f"   • Best model will be automatically identified")
    print(f"   • Use analyze_tuning_results.py to analyze results")
    
    # Save tuning job name for later analysis
    with open(f'tuning_job_{args.task}.txt', 'w') as f:
        f.write(tuner.latest_tuning_job.name)
    
    print(f"\n📝 Tuning job name saved to: tuning_job_{args.task}.txt")

if __name__ == '__main__':
    main()
