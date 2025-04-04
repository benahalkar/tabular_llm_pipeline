#!/bin/bash

# get current directory
ROOTDIR=$(dirname "$(realpath "$0")")

# enable disable wandb
online_wandb_enable=0

# get venv name
VENVNAME="tabular_llm_venv"

# get venv path
VENVROOT=${ROOTDIR}/../${VENVNAME}

# Ensure the virtual environment exists
if ! [ -d $VENVROOT ]; then
    echo "Virtual environment Directory does not exist."
    exit 1
fi


if [ "$online_wandb_enable" -eq 0 ]; then
    # disable wandb since it cannot be used on offline systems
    export WANDB_MODE=disabled

elif [ "$online_wandb_enable" -eq 1 ]; then
    # Ensure WANDB_API_KEY exists
    if [ -z "$WANDB_API_KEY" ]; then
        echo "Error: WANDB_API_KEY is not set. Please export it before running the script."
        exit 1
    fi

else
    echo "The variable is neither 0 nor 1"
fi


# get source python path
PYTHON_PATH=${VENVROOT}/bin/python 

# activate virtual environment
source ${VENVROOT}/bin/activate

# explicitly set the number of visible devices
CUDA_VISIBLE_DEVICES=0,1,2,3

# Launch script
# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0,1,2,3 ${PYTHON_PATH} ${FILE_PATH}/train.py \
# CUDA_LAUNCH_BLOCKING=1 accelerate launch ${ROOTDIR}/train.py \
CUDA_LAUNCH_BLOCKING=1 deepspeed ${FILE_PATH}/train.py \
    --deepspeed ${ROOTDIR}/zero3.json \
    --model_name_or_path meta-llama/Llama-3.2-8B \
    --data_path ${ROOTDIR}/prompt_generation \
    --attn_implementation flash_attention_2 \
    --device cuda \
    --optim adamw_8bit \
    --bits 8 \
    --lora_enable True \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_threshold 6.0 \
    --lora_bias none \
    --fp16 True \
    --bf16 False \
    --tf32 False \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --eval_strategy "steps" \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --weight_decay 0.03 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "linear" \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --output_dir ./finetuned_model \
    --logging_dir ./logs \
    --logging_steps 10 \
    --learning_rate 3e-4 \
    --load_best_model_at_end True \
    --report_to wandb

    # --model_name_or_path ai21labs/AI21-Jamba-1.5-Mini \
    # --data_path ${ROOTDIR}/data \

# disable virtual environment
deactivate

# STOP
exit 0
