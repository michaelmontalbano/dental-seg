import sagemaker
import pandas as pd
import argparse
import json
import boto3
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def get_tuning_results(tuning_job_name):
    """Extract results from a hyperparameter tuning job"""
    
    sm_client = boto3.client('sagemaker')
    
    # Get tuning job details
    tuning_job = sm_client.describe_hyper_parameter_tuning_job(
        HyperParameterTuningJobName=tuning_job_name
    )
    
    # Get all training jobs from this tuning job
    training_jobs = []
    next_token = None
    
    while True:
        if next_token:
            response = sm_client.list_training_jobs_for_hyper_parameter_tuning_job(
                HyperParameterTuningJobName=tuning_job_name,
                NextToken=next_token,
                MaxResults=100
            )
        else:
            response = sm_client.list_training_jobs_for_hyper_parameter_tuning_job(
                HyperParameterTuningJobName=tuning_job_name,
                MaxResults=100
            )
        
        training_jobs.extend(response['TrainingJobSummaries'])
        
        if 'NextToken' in response:
            next_token = response['NextToken']
        else:
            break
    
    # Extract details for each training job
    results = []
    for job_summary in training_jobs:
        job_name = job_summary['TrainingJobName']
        job_status = job_summary['TrainingJobStatus']
        
        if job_status == 'Completed':
            # Get detailed job info
            job_details = sm_client.describe_training_job(TrainingJobName=job_name)
            
            # Extract hyperparameters
            hyperparameters = job_details['HyperParameters']
            
            # Extract final metric
            final_metric = job_summary.get('FinalHyperParameterTuningJobObjectiveMetric', {})
            
            result = {
                'job_name': job_name,
                'status': job_status,
                'objective_value': final_metric.get('Value', float('inf')),
                'objective_metric': final_metric.get('MetricName', 'unknown'),
                'training_time': (job_details['TrainingEndTime'] - job_details['TrainingStartTime']).total_seconds() / 60,
                **hyperparameters
            }
            
            results.append(result)
    
    return pd.DataFrame(results)

def analyze_results(df, task):
    """Analyze hyperparameter tuning results"""
    
    print(f"\n🎯 Hyperparameter Tuning Analysis for {task.upper()} Regression")
    print("=" * 60)
    
    # Sort by objective value
    df_sorted = df.sort_values('objective_value')
    
    # Best model
    print("\n🏆 BEST MODEL:")
    best_model = df_sorted.iloc[0]
    print(f"   Job: {best_model['job_name']}")
    print(f"   {best_model['objective_metric']}: {best_model['objective_value']:.3f}")
    print(f"   Training time: {best_model['training_time']:.1f} minutes")
    
    print("\n📊 Best Hyperparameters:")
    important_params = ['learning_rate', 'batch_size', 'model_size', 'embed_dim', 
                       'unfreeze_epoch', 'full_unfreeze_epoch', 'company_lr_multiplier',
                       'weight_decay', 'use_attention']
    
    for param in important_params:
        if param in best_model:
            print(f"   {param}: {best_model[param]}")
    
    # Top 5 models
    print("\n📈 TOP 5 MODELS:")
    print("-" * 60)
    for i, row in df_sorted.head(5).iterrows():
        print(f"{i+1}. {row['objective_metric']}: {row['objective_value']:.3f} "
              f"(lr={row.get('learning_rate', 'N/A')}, "
              f"bs={row.get('batch_size', 'N/A')}, "
              f"model={row.get('model_size', 'N/A')})")
    
    # Parameter analysis
    print("\n🔍 PARAMETER IMPACT ANALYSIS:")
    print("-" * 60)
    
    # Convert string parameters to numeric where possible
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass
    
    # Analyze each parameter
    numeric_params = ['learning_rate', 'batch_size', 'embed_dim', 'unfreeze_epoch', 
                     'full_unfreeze_epoch', 'company_lr_multiplier', 'weight_decay']
    
    for param in numeric_params:
        if param in df.columns:
            try:
                # Group by parameter value and get mean objective
                grouped = df.groupby(param)['objective_value'].agg(['mean', 'std', 'count'])
                
                print(f"\n{param}:")
                print(grouped.sort_values('mean').to_string())
            except:
                pass
    
    # Categorical parameters
    categorical_params = ['model_size', 'use_attention']
    
    for param in categorical_params:
        if param in df.columns:
            grouped = df.groupby(param)['objective_value'].agg(['mean', 'std', 'count'])
            
            print(f"\n{param}:")
            print(grouped.sort_values('mean').to_string())
    
    return df_sorted

def create_visualizations(df, task, output_dir='.'):
    """Create visualization plots"""
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Parameter importance plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    params_to_plot = ['learning_rate', 'batch_size', 'embed_dim', 
                     'unfreeze_epoch', 'company_lr_multiplier', 'weight_decay']
    
    for i, param in enumerate(params_to_plot):
        if param in df.columns and i < len(axes):
            try:
                df_numeric = df[pd.to_numeric(df[param], errors='coerce').notna()]
                df_numeric[param] = pd.to_numeric(df_numeric[param])
                
                axes[i].scatter(df_numeric[param], df_numeric['objective_value'], alpha=0.6)
                axes[i].set_xlabel(param)
                axes[i].set_ylabel(f'{task}_mae')
                axes[i].set_title(f'{param} vs Objective')
                
                # Add trend line
                if len(df_numeric) > 3:
                    z = np.polyfit(df_numeric[param], df_numeric['objective_value'], 1)
                    p = np.poly1d(z)
                    axes[i].plot(df_numeric[param], p(df_numeric[param]), "r--", alpha=0.8)
            except:
                axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/parameter_analysis_{task}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Model size comparison
    if 'model_size' in df.columns:
        plt.figure(figsize=(10, 6))
        df.boxplot(column='objective_value', by='model_size')
        plt.title(f'Model Size vs {task}_mae')
        plt.ylabel(f'{task}_mae')
        plt.xlabel('Model Size')
        plt.savefig(f'{output_dir}/model_size_comparison_{task}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"\n📊 Visualizations saved to {output_dir}/")

def main():
    parser = argparse.ArgumentParser(description='Analyze hyperparameter tuning results')
    parser.add_argument('--tuning_job_name', type=str, help='Name of the tuning job')
    parser.add_argument('--tuning_job_file', type=str, help='File containing tuning job name')
    parser.add_argument('--task', type=str, default='diameter',
                        choices=['length', 'diameter', 'both'],
                        help='Regression task')
    parser.add_argument('--save_csv', type=str, help='Path to save results CSV')
    
    args = parser.parse_args()
    
    # Get tuning job name
    if args.tuning_job_name:
        tuning_job_name = args.tuning_job_name
    elif args.tuning_job_file:
        with open(args.tuning_job_file, 'r') as f:
            tuning_job_name = f.read().strip()
    else:
        # Try to read from default file
        try:
            with open(f'tuning_job_{args.task}.txt', 'r') as f:
                tuning_job_name = f.read().strip()
        except:
            print("❌ Error: Please provide tuning job name via --tuning_job_name or --tuning_job_file")
            return
    
    print(f"📊 Analyzing tuning job: {tuning_job_name}")
    
    # Get results
    try:
        df = get_tuning_results(tuning_job_name)
        
        if len(df) == 0:
            print("❌ No completed training jobs found")
            return
        
        print(f"✅ Found {len(df)} completed training jobs")
        
        # Analyze results
        df_sorted = analyze_results(df, args.task)
        
        # Create visualizations
        try:
            import numpy as np
            create_visualizations(df, args.task)
        except Exception as e:
            print(f"⚠️  Could not create visualizations: {e}")
        
        # Save results
        if args.save_csv:
            df_sorted.to_csv(args.save_csv, index=False)
            print(f"\n💾 Results saved to: {args.save_csv}")
        else:
            # Save with default name
            output_file = f'tuning_results_{args.task}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            df_sorted.to_csv(output_file, index=False)
            print(f"\n💾 Results saved to: {output_file}")
        
        # Export best hyperparameters
        best_params = df_sorted.iloc[0].to_dict()
        best_params_file = f'best_hyperparameters_{args.task}.json'
        
        # Clean up parameters for export
        export_params = {}
        for k, v in best_params.items():
            if k not in ['job_name', 'status', 'objective_value', 'objective_metric', 'training_time']:
                export_params[k] = v
        
        with open(best_params_file, 'w') as f:
            json.dump(export_params, f, indent=2)
        
        print(f"💾 Best hyperparameters saved to: {best_params_file}")
        
    except Exception as e:
        print(f"❌ Error analyzing tuning job: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
