"""
Experimental code to attack the golden resposne with BERT-retrieval model
"""
from transformers import (
    BertTokenizer,
    BertForNextSentencePrediction,
    BertConfig,
)

import os
from utils import get_logger, set_random_seed
import torch
from functools import partial
from tqdm import tqdm

import argparse
from datasets import TURN_TOKEN, NSPDataset, EvalDataset


def main():
    logger = get_logger()
    set_random_seed()
    device = torch.device("cuda")

    """
    Load Model
    """
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    special_tokens_dict = {"additional_special_tokens": ["[SEPT]"]}
    tokenizer.add_special_tokens(special_tokens_dict)
    modelconfig = BertConfig.from_pretrained("bert-base-uncased")
    model = BertForNextSentencePrediction(modelconfig)
    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load("./logs/rand_neg1/model/epoch-0.pth"))
    model.eval()
    model.to(device)

    """
    Load Dataset
    """
    train_dataset, valid_dataset = (
        NSPDataset(
            "./data/negative/random_neg1_train.txt",
            128,
            tokenizer,
            num_neg=1,
        ),
        NSPDataset(
            "./data/negative/random_neg1_valid.txt",
            128,
            tokenizer,
            num_neg=1,
        ),
    )


if __name__ == "__main__":
    main()