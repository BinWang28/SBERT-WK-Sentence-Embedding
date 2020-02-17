from __future__ import absolute_import, division, unicode_literals

import sys
import io
import numpy as np
import logging
import argparse
import torch
import random

from transformers import *
import utils

# Set PATHs
PATH_TO_SENTEVAL = './SentEval/'
PATH_TO_DATA = './SentEval/data/'
# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

# -----------------------------------------------
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

# -----------------------------------------------
# SentEval prepare and batcher
def prepare(params, samples):
    return

# -----------------------------------------------
def batcher(params, batch):

    model = params['model']
    sentences = [' '.join(s) for s in batch]

    tokenizer = params['tokenizer']
    sentences_index = [tokenizer.encode(s, add_special_tokens=True) for s in sentences]

    features_input_ids = []
    features_mask = []
    for sent_ids in sentences_index:
        # Truncate if too long
        if len(sent_ids) > params['max_seq_length']:
            sent_ids = sent_ids[:params['max_seq_length']]
        sent_mask = [1] * len(sent_ids)
        # Padding
        padding_length = params['max_seq_length'] - len(sent_ids)
        sent_ids += ([0] * padding_length)
        sent_mask += ([0] * padding_length)
        # Length Check
        assert len(sent_ids) == params['max_seq_length']
        assert len(sent_mask) == params['max_seq_length']

        features_input_ids.append(sent_ids)
        features_mask.append(sent_mask)

    batch_input_ids = torch.tensor(features_input_ids, dtype=torch.long)
    batch_input_mask = torch.tensor(features_mask, dtype=torch.long)
    batch = [batch_input_ids.to(device), batch_input_mask.to(device)]


    inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
    model.zero_grad()

    with torch.no_grad():
        features = model(**inputs)[1]

    features = [layer_emb.cpu().numpy() for layer_emb in features]
    all_layer_embedding = []
    for i in range(features[0].shape[0]):
        all_layer_embedding.append(np.array([layer_emb[i] for layer_emb in features]))

    embed_method = utils.generate_embedding(params['embed_method'], features_mask)
    embedding = embed_method.embed(params, all_layer_embedding)

    return embedding

# -----------------------------------------------



if __name__ == "__main__":
    # -----------------------------------------------
    # Settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int,
                        help="batch size for extracting features.")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--model_type", type=str, default='bert-base-uncased',
                        help="Pre-trained language models. (default: 'bert-base-uncased')")
    parser.add_argument("--embed_method", type=str, default='ave_last_hidden',
                        help="Choice of method to obtain embeddings (default: 'ave_last_hidden')")
    parser.add_argument("--context_window_size", type=int, default=2,
                        help='Topological Embedding Context Window Size (default: 2)')
    parser.add_argument("--tasks", type=str, default='all',
                        help='choice of tasks to evaluate on') 
    args = parser.parse_args()

    # -----------------------------------------------
    # Set device
    torch.cuda.set_device(-1)
    device = torch.device("cuda", 0)
    args.device = device

    # -----------------------------------------------
    # Set seed
    set_seed(args)
    # Set up logger
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

    # -----------------------------------------------
    # Set Model
    params_senteval = vars(args)

    #if params_senteval["model_type"] == 'bert-base-uncased' or 'roberta-base' or 'xlnet-base-cased':
    if 1:
        config = AutoConfig.from_pretrained(params_senteval["model_type"])
        config.output_hidden_states = True
        tokenizer = AutoTokenizer.from_pretrained(params_senteval["model_type"])
        model = AutoModelWithLMHead.from_pretrained(params_senteval["model_type"], config=config)

    params_senteval['tokenizer'] = tokenizer
    params_senteval['model'] = model
    model.to(params_senteval['device'])


    # -----------------------------------------------
    # Set params for SentEval
    params_senteval.update({'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10})
    params_senteval['classifier'] = {'nhid': 50, 'optim': 'adam', 'batch_size': 64,
                                 'tenacity': 5, 'epoch_size': 4}
    

    # ------------------- Evalution Setting ---------------------------------------
    se = senteval.engine.SE(params_senteval, batcher, prepare)

    if args.tasks == 'all':
        transfer_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'SST5',
                      'TREC', 'MRPC', 'SICKRelatedness', 'SICKEntailment', 
                      'STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark']
        transfer_tasks.extend(['Length', 'WordContent', 'Depth', 'TopConstituents',
                      'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                      'OddManOut', 'CoordinationInversion'])
    
    elif args.tasks == 'supervised':
        transfer_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'SST5',
                      'TREC', 'MRPC', 'SICKRelatedness', 'SICKEntailment']
    
    elif args.tasks == 'sts':
        transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark']

    elif args.tasks == 'probing':
        transfer_tasks = ['Length', 'WordContent', 'Depth', 'TopConstituents',
                      'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                      'OddManOut', 'CoordinationInversion']

    results = se.eval(transfer_tasks)

    # --- Print Results ---    
    for key, val in results.items():
        if key in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'SST5', 'TREC', 'SICKEntailment',
                    'Length', 'WordContent', 'Depth', 'TopConstituents',
                    'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                    'OddManOut', 'CoordinationInversion']:
            print(key, '-->', ' acc:', val['acc'])
            print('\n')
        elif key in ['MRPC']:
            print(key, '-->',' acc:', val['acc'], ' f1:', val['f1'])
            print('\n')
        elif key in ['SICKRelatedness', 'STSBenchmark']:
            print(key, '-->', ' pearson:', val['pearson'], ' spearman:', val['spearman'])
            print('\n')
        elif key in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
            print(key, '-->', ' pearson:', val['all']['pearson']['mean'], ' spearman:', val['all']['spearman']['mean'])
            print('\n')