#!/bin/bash

################## VICUNA ##################
PROMPT_VERSION=v1
MODEL_VERSION="vicuna-v1-3-7b"
################## VICUNA ##################

deepspeed llava/train/train_mem_search.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-7b-v1.3 \
    --version $PROMPT_VERSION \
    --data_path /path/to/seal_vqa_data \
    --image_folder /path/to/data \
    --vision_tower openai/clip-vit-large-patch14 \
    --mm_projector_type linear \
    --object_mm_projector_type perceiver \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-$MODEL_VERSION-linear-pretrain/mm_projector.bin \
    --pretrain_mm_perceiver_adapter ./checkpoints/llava-$MODEL_VERSION-resampler-pretrain/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir ./checkpoints/seal_vqa_7b \
    --num_train_epochs 2 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
