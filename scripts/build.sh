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
    # Define the parameters for paraphrase and basic models
    params1="-c -t 0.1 0.0 1.0 0.0 -n ${model_name}_paraphrase -g -p -s 7522"
    params2="-c -t 0.1 0.0 1.0 0.0 -n ${model_name} -g -s 7522"

    # Print the model name
    echo "Building model $model_name"

    # Call the Python scripts with the parameters for paraphrase and basic models
    python scripts/mdner.py $params1
    python scripts/mdner.py $params2
done

# Deactivate the mdner environment
conda deactivate