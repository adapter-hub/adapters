#!/bin/bash

# This script performs backward compatibility tests by comparing adapter versions of different branches.
# The goal is to check if the model output produced under the same conditions is identical between branches.
# To do this, we need to determine a directory path to save the reference output produced by the current branch.
# It's important that this directory is outside the adapters repository to remain accessible when switching branches.

# Select a directory to save the reference outputs (must be outside the repository!)
SaveDir="<the/directory/>"

# Now, determine the branch you want to compare against.
Branch=<branch/to/compare/against>

# After setting these variables, you can execute this script from the back_comp directory using the command: `sh compare.sh`


cd ..
pip install -e ".[dev]"     # # Ensure that the adapters version of the current branch is installed
cd back_comp

echo "Creating reference outputs..."
python create_outputs.py --path="$SaveDir"
cd ..

 
git checkout $Branch        # Switch branch 
pip install -e ".[dev]"     # Install the other adapter version

cd back_comp
echo "Comparing to reference outputs..."
python compare_outputs.py --path="$SaveDir"