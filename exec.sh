PYTHON_PATH=`which python`
FILE_PATH=`pwd`

export WANDB_MODE=disabled

    # --deepspeed ./zero2.json \

# CUDA_LAUNCH_BLOCKING=1 deepspeed ${FILE_PATH}/train.py \
CUDA_LAUNCH_BLOCKING=1 ${PYTHON_PATH} ${FILE_PATH}/train.py \
    --model_name_or_path ai21labs/Jamba-tiny-random \
    --data_path ./data \
    --device cuda \
    --optim adamw_torch \
    --bits 8 \
    --lora_enable True \
    --lora_r 64 \
    --lora_threshold 6.0 \
    --fp16 True \
    --bf16 False \
    --tf32 False \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
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

