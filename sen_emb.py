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


# -----------------------------------------------
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


# -----------------------------------------------

if __name__ == "__main__":
    # -----------------------------------------------
    # Settings
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size", default=64, type=int, help="batch size for extracting features."
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="bert-base-uncased",
        help="Pre-trained language models. (default: 'bert-base-uncased')",
    )
    parser.add_argument(
        "--embed_method",
        type=str,
        default="ave_last_hidden",
        help="Choice of method to obtain embeddings (default: 'ave_last_hidden')",
    )
    parser.add_argument(
        "--context_window_size",
        type=int,
        default=2,
        help="Topological Embedding Context Window Size (default: 2)",
    )
    parser.add_argument(
        "--layer_start",
        type=int,
        default=4,
        help="Starting layer for fusion (default: 4)",
    )
    parser.add_argument(
        "--tasks", type=str, default="all", help="choice of tasks to evaluate on"
    )
    args = parser.parse_args()

    # -----------------------------------------------
    # Set device
    # torch.cuda.set_device(-1)
    device = torch.device("cpu")
    args.device = device

    # -----------------------------------------------
    # Set seed
    set_seed(args)
    # Set up logger
    logging.basicConfig(format="%(asctime)s : %(message)s", level=logging.DEBUG)

    # -----------------------------------------------
    # Set Model
    params = vars(args)

    config = AutoConfig.from_pretrained(params["model_type"], cache_dir="./cache")
    config.output_hidden_states = True
    tokenizer = AutoTokenizer.from_pretrained(params["model_type"], cache_dir="./cache")
    model = AutoModelWithLMHead.from_pretrained(
        params["model_type"], config=config, cache_dir="./cache"
    )
    model.to(params["device"])

    # -----------------------------------------------

    sentence1 = input("\nEnter the first sentence: ")
    sentence2 = input("Enter the second sentence: ")

    sentences = [sentence1, sentence2]

    print("The two sentences we have are:", sentences)

    # -----------------------------------------------
    sentences_index = [tokenizer.encode(s, add_special_tokens=True) for s in sentences]
    features_input_ids = []
    features_mask = []
    for sent_ids in sentences_index:
        # Truncate if too long
        if len(sent_ids) > params["max_seq_length"]:
            sent_ids = sent_ids[: params["max_seq_length"]]
        sent_mask = [1] * len(sent_ids)
        # Padding
        padding_length = params["max_seq_length"] - len(sent_ids)
        sent_ids += [0] * padding_length
        sent_mask += [0] * padding_length
        # Length Check
        assert len(sent_ids) == params["max_seq_length"]
        assert len(sent_mask) == params["max_seq_length"]

        features_input_ids.append(sent_ids)
        features_mask.append(sent_mask)

    batch_input_ids = torch.tensor(features_input_ids, dtype=torch.long)
    batch_input_mask = torch.tensor(features_mask, dtype=torch.long)
    batch = [batch_input_ids.to(device), batch_input_mask.to(device)]

    inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
    model.zero_grad()

    with torch.no_grad():
        features = model(**inputs)[1]

    # Reshape features from list of (batch_size, seq_len, hidden_dim) for each hidden state to list
    # of (num_hidden_states, seq_len, hidden_dim) for each element in the batch.
    all_layer_embedding = list(
        np.array([hidden_state.cpu().numpy() for hidden_state in features]).swapaxes(
            0, 1
        )
    )

    embed_method = utils.generate_embedding(params["embed_method"], features_mask)
    embedding = embed_method.embed(params, all_layer_embedding)

    similarity = (
        embedding[0].dot(embedding[1])
        / np.linalg.norm(embedding[0])
        / np.linalg.norm(embedding[1])
    )
    print("The similarity between these two sentences are (from 0-1):", similarity)
