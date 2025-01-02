PYTHON_PATH=`which python`
FILE_PATH=`pwd`

export WANDB_MODE=disabled

# CUDA_LAUNCH_BLOCKING=1 deepspeed ${FILE_PATH}/train.py \
CUDA_LAUNCH_BLOCKING=1 ${PYTHON_PATH} ${FILE_PATH}/train.py \
    --deepspeed ./zero2.json \
    --model_name_or_path ai21labs/Jamba-tiny-random \
    --data_path ./data \
    --bits 8 \
    --lora_enable True \
    --fp16 False \
    --bf16 False \
    --threshold 6.0 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --output_dir ./finetuned_model \
    --device cuda

