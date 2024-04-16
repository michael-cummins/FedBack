#!/bin/bash

# Path to the directory containing .sub files
sub_dir="event_subs/mnist"

# Loop through each .sub file in the directory and submit them
for sub_file in $sub_dir/*.sub; do
    if [ -f "$sub_file" ]; then
        condor_submit_bid 50 "$sub_file"
    fi
done

# Path to the directory containing .sub files
sub_dir="event_subs/cifar"

# Loop through each .sub file in the directory and submit them
for sub_file in $sub_dir/*.sub; do
    if [ -f "$sub_file" ]; then
        condor_submit_bid 200 "$sub_file"
    fi
done