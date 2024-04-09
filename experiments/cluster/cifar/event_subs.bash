#!/bin/bash

# Path to the directory containing .sub files
sub_dir="event_subs"

# Loop through each .sub file in the directory and submit them
for sub_file in $sub_dir/*.sub; do
    if [ -f "$sub_file" ]; then
        condor_submit_bid 500 "$sub_file"
    fi
done