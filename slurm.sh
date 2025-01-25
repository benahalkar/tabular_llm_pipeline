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

# get current directory
ROOTDIR=$(dirname "$(realpath "$0")")

# Load modules or conda environment
# module load python/3.8
module load python/3.11
# module load cuda/11.3
module load cuda

# define the execution script name and execute
SCRIPT_NAME=${ROOTDIR}/exec.sh
/usr/bin/bash ${SCRIPT_NAME}

# STOP
exit 0