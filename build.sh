#!/bin/bash

# define the directory name for the github repo
DIRNAME="tabular_llm_pipeline"

# define the directory name for the virtual environment
VENVNAME="tabular_llm_venv"

# Ensure GITHUB_API_KEY exists
if [ -z "$GITHUB_API_KEY" ]; then
    echo "Error: GITHUB_API_KEY is not set. Please export it before running the script."
    exit 1
fi

# Ensure Python 3 is used
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH."
    exit 1
fi

# clone the repository
git clone https://${GITHUB_KEY}@github.com/benahalkar/tabular_llm_pipeline.git ${DIRNAME}

# create the virtual environment
python -m venv ${VENVNAME}

# activate virtual environment
source ${VENVNAME}/bin/activate

# move into repository
cd ${DIRNAME} 

# download all packages
pip install -r requirements.txt

# move out of repository
cd ..

# Configure accelerate
echo "Configuring accelerate now"
accelerate config

# disable virtual environment
deactivate

# user message
echo "All setup and download completed!"

# STOP
exit 0