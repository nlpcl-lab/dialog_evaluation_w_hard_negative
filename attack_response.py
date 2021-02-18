"""
Experimental code to attack the golden resposne with BERT-retrieval model
"""
import argparse
import os
from functools import partial

import torch
from tqdm import tqdm
from transformers import (BertConfig, BertForNextSentencePrediction,
                          BertTokenizer)

from datasets import TURN_TOKEN, EvalDataset, NSPDataset
from utils import get_logger, read_raw_file, set_random_seed


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
    train_raw, valid_raw = (
        read_raw_file("./data/negative/random_neg1_train.txt"),
        read_raw_file("./data/negative/random_neg1_valid.txt"),
    )

    for conversation in valid_raw:
        context, response, _ = conversation
        original_encoded = tokenizer(
            context,
            text_pair=response,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        original_score = get_nsp_score(original_encoded, model, device)
        print(original_score)


def get_nsp_score(sample, model, device) -> float:
    """
    input_ids~token_type_ids는 (1, max_seq_len)의 shape을 가짐.
    """
    input_ids, attention_masks, token_type_ids = (
        sample["input_ids"],
        sample["attention_mask"],
        sample["token_type_ids"],
    )
    softmax = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        output = (
            softmax(
                model(
                    input_ids.to(device),
                    attention_masks.to(device),
                    token_type_ids.to(device),
                )[0]
            )
            .cpu()
            .numpy()[0][0]
        )
    return float(output)


if __name__ == "__main__":
    main()