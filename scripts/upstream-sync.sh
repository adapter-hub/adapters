#!/usr/bin/env bash
#
# A little script that helps with merging upstream HF changes into adapter-transformers.
# The script should be called from the repo root, e.g.:
# ./scripts/upstream-sync.sh v4.9.2
#
# It requires "git-strip-merge" placed in the same folder, which can be obtained from here (not included for licensing reasons):
# https://github.com/MestreLion/git-tools/blob/main/git-strip-merge
#

TAG=$1
BRANCH_NAME=sync/$1

DELETE_PATHS=(
    ".circleci/*"
    ".github/*"
    "docs/*"
    "examples/flax/*"
    "examples/legacy/*"
    "examples/research_projects/*"
    "examples/tensorflow/*"
    "model_cards/*"
    "notebooks/*"
    "scripts/*"
    "templates/*"
    "README_*.md"
)

IGNORE_PATHS=(
    ".github/*"
    "adapter_docs/*"
    "notebooks/*"
    "scripts/*"
    "CITATION.cff"
    "README.md"
    "CONTRIBUTING.md"
)

git fetch upstream --prune
git branch $BRANCH_NAME
git checkout $BRANCH_NAME
# Merge & strip not required files
./scripts/git-strip-merge $1 "${DELETE_PATHS[@]}"
# Reset all files where we always keep our version
for path in "${IGNORE_PATHS[@]}"; do
    git checkout HEAD -- "$path"
    git add -u "$path"
done
