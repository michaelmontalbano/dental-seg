import sagemaker
from sagemaker.estimator import Estimator
from datetime import datetime

# --- Configuration ---
image_uri = "552401576656.dkr.ecr.us-west-2.amazonaws.com/nnunet-dental-segmentation:latest-cuda12.1-py310"

role = "arn:aws:iam::552401576656:role/SageMakerExecutionRole"
bucket = "codentist-general"
s3_root_uri = "s3://codentist-general/datasets/nnUNet_raw"
job_name = f"nnunet-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

# --- SageMaker session ---
session = sagemaker.Session()

# --- Input Channels ---
inputs = {
    "nnUNet_raw": s3_root_uri  # Contains Dataset300_BoneLoss inside
}
# --- Estimator ---
estimator = Estimator(
    image_uri=image_uri,
    role=role,
    instance_count=1,
    instance_type="ml.g4dn.2xlarge",
    volume_size=100,
    max_run=60 * 60 * 12,
    input_mode="File",
    output_path=f"s3://{bucket}/nnunet-output",
    base_job_name=job_name,
    hyperparameters={
        "task-name": "Dataset300_BoneLoss",
        "dataset-id": 300,
        "configuration": "3d_fullres",
        "folds": "0",  # You can also do "0 1 2 3 4"
        "trainer-class-name": "nnUNetTrainer",
        "plans-name": "nnUNetPlans",
        "preprocessor-name": "DentalCBCTPreprocessor",
        "num-gpus": 1,
        "num-workers-dataloader": 8,
        "gpu-memory-target": 8,
        "export-validation-probabilities": True,
        "val-with-best": True
    },
    environment={
        "nnUNet_raw_data_base": "/opt/ml/input/data/nnUNet_raw",
        "nnUNet_preprocessed": "/opt/ml/input/data/nnUNet_preprocessed",
        "RESULTS_FOLDER": "/opt/ml/model"
    }
)


# --- Launch ---
estimator.fit(inputs, wait=True)

print(f"✅ SageMaker training job launched: {job_name}")