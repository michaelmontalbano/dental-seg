#!/bin/bash
INSTANCE_ID="i-06062cbfcf76dae1d"
REGION="us-west-2"

case "$1" in
    start)
        echo "▶️  Starting instance..."
        aws ec2 start-instances --instance-ids $INSTANCE_ID --region $REGION
        aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $REGION
        PUBLIC_IP=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --region $REGION             --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
        echo "✅ Instance started. IP: $PUBLIC_IP"
        # Update connect script
        sed -i.bak "s/ec2-user@.*/ec2-user@$PUBLIC_IP/" connect_to_builder.sh
        ;;
    stop)
        echo "⏹️  Stopping instance..."
        aws ec2 stop-instances --instance-ids $INSTANCE_ID --region $REGION
        aws ec2 wait instance-stopped --instance-ids $INSTANCE_ID --region $REGION
        echo "✅ Instance stopped"
        ;;
    status)
        STATE=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --region $REGION             --query 'Reservations[0].Instances[0].State.Name' --output text)
        echo "📊 Instance state: $STATE"
        if [ "$STATE" == "running" ]; then
            PUBLIC_IP=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --region $REGION                 --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
            echo "🌐 Public IP: $PUBLIC_IP"
        fi
        ;;
    terminate)
        read -p "⚠️  Are you sure you want to terminate the instance? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "🗑️  Terminating instance..."
            aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION
            echo "✅ Instance terminated"
        fi
        ;;
    *)
        echo "Usage: $0 {start|stop|status|terminate}"
        exit 1
        ;;
esac
