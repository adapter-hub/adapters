# Backwards Compatibility Tests

## Motivation

This directory contains a set of tests that can be run to ensure newly introduced changes or performed refactorings don't break the existing functionalities.
We assume that the changes are developed on a separate branch, e.g., dev`, checked out from the main branch, `main`.

The script performs a forward pass for each supported model and compares if the outputs differ on `main` and `dev` or is identical. 

## Requirements
For executing these tests, certain requirements must be met:
- The ability to execute bash scripts (built-in under Linux/macOS; for Windows, we refer to using third-party software, such as [GNU](https://www.gnu.org/software/bash/))
- Git as a version control system to be able to checkout branches
- The ability to checkout to the desired branch; if the branch is, e.g., from another fork you may need to add the repository as a remote (see [here](https://docs.github.com/en/get-started/getting-started-with-git/managing-remote-repositories) for instructions)
- A virtual environment with Python to modify the package version installed of `adapters`

## Procedure

1. Determine a directory to save the model output produced by the test and save the path to the variable `SaveDir` in the shell script compare.sh
2. Select the branch you want to compare to `main` and save it to the variable `Branch`
3. In your command line, navigate to the `back_comp` dir and execute the script by typing `sh compare.sh`

The results are visualized in the command line.