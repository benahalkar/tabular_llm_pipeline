#!/bin/bash

VENV_PATH="/home/harshbenahalkar/ra_venv"


if ! [ -d $VENV_PATH ]; then
    echo "Directory does not exist."
    exit 1
fi

PYTHON_PATH="$VENV_PATH/bin/" 
FILE_PATH=$(dirname "$(realpath "$0")")

export WANDB_MODE=disabled

    # --deepspeed ./zero2.json \
# CUDA_LAUNCH_BLOCKING=1 deepspeed ${FILE_PATH}/train.py \
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0,1,2,3 ${PYTHON_PATH}python ${FILE_PATH}/train.py \
    --model_name_or_path ai21labs/Jamba-tiny-random \
    --data_path data \
    --device cuda \
    --optim adamw_torch \
    --bits 4 \
    --lora_enable True \
    --lora_r 32 \
    --lora_threshold 6.0 \
    --fp16 True \
    --bf16 False \
    --tf32 False \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "steps" \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --output_dir ./finetuned_model \
    --logging_dir ./logs \
    --logging_steps 10 \
    --learning_rate 2e-3 \
    --load_best_model_at_end True \
    --report_to wandb

exit 0