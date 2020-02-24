#!/usr/bin/env bash


args=(
    --model_type 'binwang/bert-base-nli' 		
    	# Model Selection

    --embed_method 'ave_last_hidden'     		
    	# 'ave_last_hidden'
    	# 'dissecting'
    	# 'CLS'
    	# 'ave_one_layer'

    --max_seq_length 128                
    	# Maximum length for a sentence

    --batch_size 64                     
    	# Batch size for extracting sentence features

    --context_window_size 2             
    	# Window size for Alignment and Novelty
    
    --layer_start 4 					
    	# starting layer for 'dissecting' mode
    	# layer selection for 'ave_one_layer' mode

    --tasks 'all'
    	# 'sts' for sts tasks
    	# 'supervised' for supervised tasks
    	# 'probing' for probing tasks
    	# 'all' for all tasks

    )

echo "${args[@]}"
echo ""

python SBERT_WK.py "${args[@]}"