#!/bin/bash

# Launch 5 different 2-stage training jobs on ml.g6.24xlarge instances
# Each command runs with --no-wait to submit jobs without blocking

echo "🚀 Launching 5 yolo2stage training jobs on ml.g6.24xlarge instances..."

# Job 1: Bone loss detection
# echo "📊 Submitting Job 1: Bone loss detection..."
# python sagemaker_launcher.py \
#   --class-group surfaces \
#   --instance-type ml.g6.24xlarge \
#   --no-wait

# # Job 2: Decay detection
# echo "📊 Submitting Job 2: Decay detection..."
# python sagemaker_launcher.py \
#   --class-group bone-loss \
#   --instance-type ml.g6.24xlarge \
#   --no-wait

# Job 3: Enamel detection
echo "📊 Submitting Job 3: Enamel detection..."
python sagemaker_launcher.py \
  --class-group parl-rct-pulp-root \
  --instance-type ml.g6.24xlarge \
  --no-wait

# Job 4: CEJ detection
echo "📊 Submitting Job 4: CEJ detection..."
python sagemaker_launcher.py \
  --class-group decay-enamel-pulp-crown \
  --instance-type ml.g6.24xlarge \
  --no-wait

# Job 5: Apex detection
echo "📊 Submitting Job 5: Apex detection..."
python sagemaker_launcher.py \
  --class-group all-conditions \
  --instance-type ml.g6.24xlarge \
  --no-wait

echo "✅ All 5 training jobs have been submitted!"
echo "🖥️  Using ml.g6.24xlarge instances (1x NVIDIA L4 GPU, 24GB VRAM)"
echo ""
echo "📊 Monitor jobs in SageMaker console or use:"
echo "   aws sagemaker list-training-jobs --status-equals InProgress"
