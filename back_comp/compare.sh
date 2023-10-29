#!/bin/bash

# Select a directory to save the reference outputs
TestDir="/home/timo/workbench_l/AdapterHub/test/back_comp"
SaveDir="/home/timo/workbench_l/AdapterHub/test"
RepoDir="/home/timo/homeworkbench_l/AdapterHub/adapter-transformers"

cd ..
pip install -e ".[dev]"
cd $TestDir
pwd

echo "Creating reference outputs..."
python create_outputs.py --path="$SaveDir"

cd $RepoDir
pwd

git checkout clifton/dev/x-adapters 
pip install -e ".[dev]"
cd $TestDir
pwd

echo "Comparing to reference outputs..."
python compare_outputs.py --path="$SaveDir"