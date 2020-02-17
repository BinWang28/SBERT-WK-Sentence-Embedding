#!/usr/bin/env bash


args=(
    --model_type 'binwang/bert-base-nli'
    --embed_method 'dissecting'    		
    --max_seq_length 128                # Truncate if exceeding max length
    --batch_size 64                     # Batch size for extracting sentence features
    --context_window_size 2             # Window size for Alignment and Novelty
    --tasks 'sts'
    )

echo "${args[@]}"
echo ""

python SBERT_WK.py "${args[@]}"