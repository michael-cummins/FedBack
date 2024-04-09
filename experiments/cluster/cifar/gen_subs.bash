#!/bin/bash

# Define a list of rates
rates=(5 10 15 20 30 40 50 60 70 80 90 100)

# Loop through each rate in the list and generate .sub files
for rate in "${rates[@]}"; do
    python sub_generator.py --avg --rate "$rate"  # For avg
    python sub_generator.py --prox --rate "$rate" # For prox
    python sub_generator.py --admm --rate "$rate" # For admm
    python sub_generator.py --back --rate "$rate" # For admm
done
