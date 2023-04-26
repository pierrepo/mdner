#!/bin/bash
# Configurations of the conda environment
source /home/mohamed/miniconda3/etc/profile.d/conda.sh
conda activate mdner

filename=$1
script_path="scripts/mdner.py"
# Set space as the delimiter
IFS=' '
while read line; do
# Reading each line
read -a strarr <<< "$line"
python "${script_path}" -c -t "${strarr[0]}" "${strarr[1]}" "${strarr[2]}" "${strarr[3]}"
done < $filename
