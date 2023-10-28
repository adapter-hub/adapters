#!/bin/bash

# Select a directory to save the reference outputs
SaveDir="/home/timo/workbench_l/AdapterHub/test"

echo "Creating reference outputs..."
python create_outputs.py --path="$SaveDir"

echo "Comparing to reference outputs..."
python compare_outputs.py --path="$SaveDir"