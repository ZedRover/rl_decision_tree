#!/bin/bash

# Define an array of dataset names
datas=("Spambase" "body" "seeds" "small_toy" "glass" "cncret")  # Replace with your actual dataset names

# Loop over each data name and run train.py in parallel
for data_name in "${datas[@]}"; do
    echo "Starting training for data_name=$data_name"
    python train.py --data_name="$data_name" --max_depth=2 &
done

# Wait for all background processes to complete
wait
echo "All training processes completed."
