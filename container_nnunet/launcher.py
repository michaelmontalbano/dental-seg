from sagemaker.estimator import Estimator

estimator = Estimator(
    image_uri='552401576656.dkr.ecr.us-east-1.amazonaws.com/medmamba:latest',
    role='arn:aws:iam::552401576656:role/SageMakerExecutionRole',
    instance_type='ml.p3.2xlarge',
    instance_count=1,
    volume_size=100,
    max_run=36000,
    base_job_name='medmamba-train',
    hyperparameters={
        "epochs": 10,
        "batch_size": 32
    },
    output_path='s3://codentist-general/output/medmamba-training-output',
)

estimator.fit({'train': 's3://codentist-general/datasets/implants_cls_dataset/'})

