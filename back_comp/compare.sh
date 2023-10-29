#!/bin/bash

# Select a directory to save the reference outputs
SaveDir="/home/timo/workbench_l/AdapterHub/test"

echo "Creating reference outputs..."
python create_outputs.py --path="$SaveDir"

git checkout clifton/dev/x-adapters 
ls

echo "Comparing to reference outputs..."
python compare_outputs.py --path="$SaveDir"