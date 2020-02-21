#!/usr/bin/env bash


args=(
    --model_type 'binwang/bert-base-uncased'
    --embed_method 'dissecting'     # ave_last_hidden    dissecting		
    --max_seq_length 128                # Truncate if exceeding max length
    --batch_size 64                     # Batch size for extracting sentence features
    --context_window_size 2             # Window size for Alignment and Novelty
    --tasks 'all'
    )

echo "${args[@]}"
echo ""

python SBERT_WK.py "${args[@]}"