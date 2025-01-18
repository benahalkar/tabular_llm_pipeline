#!/bin/bash

#SBATCH --job-name=table_qa_job           # Job name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks-per-node=1               # Number of tasks per node (number of GPUs)
#SBATCH --cpus-per-task=1                 # Number of CPU cores per task
#SBATCH --gres=gpu:4                      # Number of GPUs per node
#SBATCH --time=24:00:00                   # Maximum runtime (HH:MM:SS)
#SBATCH --partition=compute               # Partition name
#SBATCH --output=logs/%x_%j.out           # Standard output log (%x=job-name, %j=job-id)
#SBATCH --error=logs/%x_%j.err            # Standard error log



# get directory of repository
DIRNAME="tabular_llm_pipeline"

# get virtual environment name 
VENVNAME="tabular_llm_venv"

# get current directory
ROOTDIR=FILE_PATH=$(dirname "$(realpath "$0")")



# Load modules or conda environment
# module load python/3.8
module load python/3.11
# module load cuda/11.3
module load cuda



# get directory of virtual environment
VENV_PATH=${ROOTDIR}/${VENVNAME}



# Ensure the virtual environment exists
if ! [ -d $VENV_PATH ]; then
    echo "Virtual environment Directory does not exist."
    exit 1
fi

# Ensure WANDB_API_KEY exists
if [ -z "$WANDB_API_KEY" ]; then
    echo "Error: WANDB_API_KEY is not set. Please export it before running the script."
    exit 1
fi



# get source python path
PYTHON_PATH="$VENV_PATH/bin/python" 

# activate virtual environment
source ${VENV_PATH}/bin/activate

# define the execution script name and execute
SCRIPT_NAME=${ROOTDIR}/${DIRNAME}/exec.sh
/usr/bin/bash ${SCRIPT_NAME}

# disable virtual environment
deactivate

# STOP
exit 0