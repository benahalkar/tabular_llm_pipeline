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

# first experiment to check if the number of GPU is the case
# testing first with 8 GPUs and then with 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
CUDA_LAUNCH_BLOCKING=1 deepspeed ${FILE_PATH}/train.py \
    --deepspeed ./zero3.json \
    --model_name_or_path /home/jovyan/models/Llama-3.2-1B-Instruct \
    --data_path ./prompt_generation \
    --device cuda \
    --optim adamw_4bit \
    --bits 4 \
    --lora_enable True \
    --lora_r 4 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_threshold 6.0 \
    --lora_bias none \
    --fp16 True \
    --bf16 False \
    --tf32 False \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "steps" \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --weight_decay 0.03 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "linear" \
    --gradient_checkpointing True \
    --use_reentrant False \
    --dataloader_num_workers 4 \
    --output_dir ./finetuned_model \
    --logging_dir ./logs \
    --logging_steps 10 \
    --learning_rate 3e-4 \
    --load_best_model_at_end True \
    --report_to wandb

CUDA_VISIBLE_DEVICES=0,1,2,3
CUDA_LAUNCH_BLOCKING=1 deepspeed ${FILE_PATH}/train.py \
    --deepspeed ./zero3.json \
    --model_name_or_path /home/jovyan/models/Llama-3.2-1B-Instruct \
    --data_path ./prompt_generation \
    --device cuda \
    --optim adamw_4bit \
    --bits 4 \
    --lora_enable True \
    --lora_r 4 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_threshold 6.0 \
    --lora_bias none \
    --fp16 True \
    --bf16 False \
    --tf32 False \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "steps" \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --weight_decay 0.03 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "linear" \
    --gradient_checkpointing True \
    --use_reentrant False \
    --dataloader_num_workers 4 \
    --output_dir ./finetuned_model \
    --logging_dir ./logs \
    --logging_steps 10 \
    --learning_rate 3e-4 \
    --load_best_model_at_end True \
    --report_to wandb

# first experiment to check if the optimizers is the source of problem
# testing first with a fused optimizer and then with normal optimizer 
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
CUDA_LAUNCH_BLOCKING=1 deepspeed ${FILE_PATH}/train.py \
    --deepspeed ./zero3.json \
    --model_name_or_path /home/jovyan/models/Llama-3.2-1B-Instruct \
    --data_path ./prompt_generation \
    --device cuda \
    --optim adamw_torch_fused \
    --bits 4 \
    --lora_enable True \
    --lora_r 4 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_threshold 6.0 \
    --lora_bias none \
    --fp16 True \
    --bf16 False \
    --tf32 False \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "steps" \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --weight_decay 0.03 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "linear" \
    --gradient_checkpointing True \
    --use_reentrant False \
    --dataloader_num_workers 4 \
    --output_dir ./finetuned_model \
    --logging_dir ./logs \
    --logging_steps 10 \
    --learning_rate 3e-4 \
    --load_best_model_at_end True \
    --report_to wandb

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
CUDA_LAUNCH_BLOCKING=1 deepspeed ${FILE_PATH}/train.py \
    --deepspeed ./zero3.json \
    --model_name_or_path /home/jovyan/models/Llama-3.2-1B-Instruct \
    --data_path ./prompt_generation \
    --device cuda \
    --optim adamw_torch \
    --bits 4 \
    --lora_enable True \
    --lora_r 4 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_threshold 6.0 \
    --lora_bias none \
    --fp16 True \
    --bf16 False \
    --tf32 False \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "steps" \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --weight_decay 0.03 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "linear" \
    --gradient_checkpointing True \
    --use_reentrant False \
    --dataloader_num_workers 4 \
    --output_dir ./finetuned_model \
    --logging_dir ./logs \
    --logging_steps 10 \
    --learning_rate 3e-4 \
    --load_best_model_at_end True \
    --report_to wandb

# disable virtual environment
deactivate

# STOP
exit 0
