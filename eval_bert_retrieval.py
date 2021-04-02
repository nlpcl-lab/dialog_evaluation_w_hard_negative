import argparse
import os

import torch
from tensorboardX import SummaryWriter
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm import tqdm
from transformers import (
    BertConfig,
    BertForNextSentencePrediction,
    BertTokenizer,
)

from get_dataset import get_dd_corpus, get_zhao_dataset
from trainer import Trainer
from utils import (
    dump_config,
    get_logger,
    load_model,
    save_model,
    eval_by_NSP,
    set_random_seed,
    get_correlation,
    write_summary,
)

from datasets import TURN_TOKEN, NSPDataset, EvalDataset


def main():
    EXP_NAME = (
        "./logs_wo_ttype/k5_maxchange0.4_minchange0.1_NSPCUT0.4/model/epoch-0.pth"
        # "./logs_wo_ttype/random_neg1/model/epoch-1.pth"
    )
    set_random_seed(42)
    logger = get_logger()

    device = torch.device("cuda")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    special_tokens_dict = {"additional_special_tokens": [TURN_TOKEN]}
    tokenizer.add_special_tokens(special_tokens_dict)

    model = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")
    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load(EXP_NAME))
    model.eval()
    model.to(device)

    dataset_for_correlation = EvalDataset(
        get_zhao_dataset("dd"),
        128,
        tokenizer,
    )
    model.eval()
    result = eval_by_NSP(
        dataset_for_correlation, model, device, is_rank=False
    )
    human_score_list = []
    nsp_list = []

    for el in result:
        human_score_list.append(el["human_score"])
        nsp_list.append(el["nsp"])

    correlation = get_correlation(human_score_list, nsp_list)
    print()
    print(EXP_NAME)
    print(correlation)


if __name__ == "__main__":
    main()
