#!/bin/bash

# Launch 5 different models with different class groups
# Each command runs with --no-wait to submit jobs without blocking

echo "🚀 Launching 5 yolo2stage training jobs on ml.g6.8xlarge instances..."

# Job 1: Bone loss detection
python sagemaker_launcher.py \
  --class-group bone-loss \
  --instance-type ml.g6.8xlarge \
  --no-wait &

# Job 2: Decay detection
python sagemaker_launcher.py \
  --class-group decay \
  --instance-type ml.g6.8xlarge \
  --no-wait &

# Job 3: Enamel detection
python sagemaker_launcher.py \
  --class-group decay-enamel-pulp- \
  --instance-type ml.g6.8xlarge \
  --no-wait &

# Job 4: CEJ detection
python sagemaker_launcher.py \
  --class-group dental-work \
  --instance-type ml.g6.8xlarge \
  --no-wait &

# Job 5: Apex detection
python sagemaker_launcher.py \
  --class-group parl-rct-pulp-root \
  --instance-type ml.g6.8xlarge \
  --no-wait &

# Wait for all background jobs to complete
wait

echo "✅ All 5 training jobs have been submitted!"
echo "🖥️  Using ml.g6.8xlarge instances (1x NVIDIA L4 GPU, 24GB VRAM)"
