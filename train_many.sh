#!/bin/bash

CONFIG_DIR="./Configs"  # Directory containing config files
S3_BUCKET="s3://jennai-data/tony"  # S3 bucket base path

for CONFIG in "$CONFIG_DIR"/config_ljspeech_yt.yml; do
    # Extract log_dir from the YAML file
    LOG_DIR=$(grep "log_dir:" "$CONFIG" | awk '{print $2}' | tr -d '"')

    echo "Starting experiment with config: $CONFIG"
    echo "Logs will be saved to: $LOG_DIR"

    # Run training in the background
    mkdir -p "$LOG_DIR"
    python train_finetune.py --config_path "$CONFIG" > "$LOG_DIR/finetune_log.out" 2>&1

    echo "Experiment completed. Uploading logs to S3..."

    # Upload logs to S3
    aws s3 cp --recursive "$LOG_DIR" "$S3_BUCKET/$LOG_DIR"

    rm -rf "$LOG_DIR"

    echo "Logs uploaded and removed. Moving to next experiment..."
done

echo "All experiments completed."

