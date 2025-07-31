import argparse
import boto3
import pandas as pd
from datetime import datetime

def get_tuning_job_results(tuning_job_name):
    """Get results from a hyperparameter tuning job"""
    client = boto3.client('sagemaker')
    
    # Get the tuning job description
    response = client.describe_hyper_parameter_tuning_job(
        HyperParameterTuningJobName=tuning_job_name
    )
    
    # Get all training jobs
    training_jobs = []
    paginator = client.get_paginator('list_training_jobs_for_hyper_parameter_tuning_job')
    
    for page in paginator.paginate(HyperParameterTuningJobName=tuning_job_name):
        training_jobs.extend(page['TrainingJobSummaries'])
    
    # Extract relevant information
    results = []
    for job in training_jobs:
        if job['TrainingJobStatus'] == 'Completed':
            job_name = job['TrainingJobName']
            
            # Get detailed job info
            job_details = client.describe_training_job(TrainingJobName=job_name)
            
            # Extract hyperparameters
            hyperparams = job_details['HyperParameters']
            
            # Extract metrics
            metrics = job['FinalHyperParameterTuningJobObjectiveMetric']
            
            result = {
                'job_name': job_name,
                'objective_metric': metrics['MetricName'],
                'objective_value': metrics['Value'],
                'status': job['TrainingJobStatus'],
                **hyperparams
            }
            
            results.append(result)
    
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description='Analyze HPO results')
    parser.add_argument('--tuning-job-name', type=str, required=True,
                        help='Name of the tuning job')
    parser.add_argument('--top-k', type=int, default=10,
                        help='Number of top results to show')
    args = parser.parse_args()
    
    print(f"🔍 Analyzing tuning job: {args.tuning_job_name}")
    
    # Get results
    df = get_tuning_job_results(args.tuning_job_name)
    
    if df.empty:
        print("❌ No completed jobs found!")
        return
    
    # Sort by objective value (higher is better for accuracy)
    df_sorted = df.sort_values('objective_value', ascending=False)
    
    print(f"\n📊 Total completed jobs: {len(df)}")
    print(f"\n🏆 Top {args.top_k} results:")
    print("=" * 80)
    
    # Display top results
    for idx, row in df_sorted.head(args.top_k).iterrows():
        print(f"\n#{idx+1} - {row['objective_metric']}: {row['objective_value']:.4f}")
        print(f"   Model: {row.get('model_size', 'N/A')}")
        print(f"   Batch Size: {row.get('batch_size', 'N/A')}")
        print(f"   Learning Rate: {row.get('learning_rate', 'N/A')}")
        print(f"   Weight Decay: {row.get('weight_decay', 'N/A')}")
        print(f"   Epochs: {row.get('epochs', 'N/A')}")
        print(f"   Warmup Epochs: {row.get('warmup_epochs', 'N/A')}")
    
    # Save full results
    output_file = f"hpo_results_{args.tuning_job_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df_sorted.to_csv(output_file, index=False)
    print(f"\n💾 Full results saved to: {output_file}")
    
    # Show parameter importance
    print("\n📈 Parameter Analysis:")
    print("=" * 40)
    
    # Group by each hyperparameter and show average performance
    for param in ['model_size', 'batch_size', 'learning_rate', 'weight_decay', 'warmup_epochs']:
        if param in df.columns:
            avg_by_param = df.groupby(param)['objective_value'].agg(['mean', 'std', 'count'])
            print(f"\n{param}:")
            print(avg_by_param.sort_values('mean', ascending=False))

if __name__ == '__main__':
    main()
