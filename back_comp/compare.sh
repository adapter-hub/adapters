#!/bin/bash

# Select a directory to save the reference outputs
TestDir="/home/timo/workbench_l/AdapterHub/test/back_comp"
SaveDir="/home/timo/workbench_l/AdapterHub/test"

echo "Creating reference outputs..."
#python create_outputs.py --path="$SaveDir"

git checkout clifton/dev/x-adapters 
pwd 
cd ..
pwd

echo "Comparing to reference outputs..."
#python compare_outputs.py --path="$SaveDir"