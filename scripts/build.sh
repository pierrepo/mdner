#!/bin/bash

# Activate the mdner environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mdner

# Define the list of model names
model_names=()
for i in {1..10}
do
    model_names+=("model$i")
done

for model_name in "${model_names[@]}"
do
    # Define the parameters for each model
    params="-c -t 0.1 0.0 1.0 0.0 -n $model_name -g -p -s 7522"

    # Print the model name and parameters
    echo "Building model $model_name"

    # Call the Python script with the parameters
    python scripts/mdner.py $params
done

# Deactivate the mdner environment
conda deactivate