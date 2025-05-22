#!/bin/bash
# Simple script to run TALT experiments in Google Colab

# Setup the environment
python setup_colab.py

# Check if a config file was provided
if [ $# -eq 0 ]; then
    echo "No config file specified. Using default cnn_comparison.json"
    CONFIG_FILE="talt_evaluation/batch_configs/cnn_comparison.json"
else
    CONFIG_FILE=$1
fi

# Run the batch experiment
python talt_evaluation/run_batch.py \
    --config $CONFIG_FILE \
    --output-dir ./results