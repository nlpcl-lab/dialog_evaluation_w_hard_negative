import os
import json

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


def scoring_main():
    device = torch.device("cuda")
    fnames = "./attack/neg2_valid_k5_maxchange0.4_minchange0.1_NSPCUT0.4.txt"

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    special_tokens_dict = {"additional_special_tokens": [TURN_TOKEN]}
    tokenizer.add_special_tokens(special_tokens_dict)

    model = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")
    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load("./logs_wo_ttype/random_neg1/model/epoch-0.pth"))
    model.eval()
    model.to(device)
    softmax = torch.nn.Softmax(dim=1)

    golden_score_list, random_score_list, ours_score_list = [], [], []
    with open(fnames, "r") as f:
        ls = [el.strip().split("|||") for el in f.readlines()]
    for line in tqdm(ls):
        context, golden, rand_neg, ours_neg = line
        if ours_neg == "[NONE]":
            continue
        encoded = tokenizer(
            context,
            text_pair=golden,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            prediction = (
                softmax(
                    model(encoded["input_ids"].to(device), encoded["attention_mask"].to(device))[0]
                )
                .cpu()
                .numpy()[0][0]
            )
            golden_score_list.append(prediction)

        encoded = tokenizer(
            context,
            text_pair=rand_neg,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            prediction = (
                softmax(
                    model(encoded["input_ids"].to(device), encoded["attention_mask"].to(device))[0]
                )
                .cpu()
                .numpy()[0][0]
            )
            random_score_list.append(prediction)

        encoded = tokenizer(
            context,
            text_pair=ours_neg,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            prediction = (
                softmax(
                    model(encoded["input_ids"].to(device), encoded["attention_mask"].to(device))[0]
                )
                .cpu()
                .numpy()[0][0]
            )
            ours_score_list.append(prediction)

    assert len(ours_score_list) == len(golden_score_list) == len(random_score_list)
    with open("prediction_list.json", "w") as f:
        json.dump(
            {
                "golden": [str(el) for el in golden_score_list],
                "ours": [str(el) for el in ours_score_list],
                "random": [str(el) for el in random_score_list],
            },
            f,
        )


def boxing_main():
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    labelsize = 16
    rcParams["xtick.labelsize"] = labelsize
    rcParams["ytick.labelsize"] = labelsize

    with open("prediction_list.json", "r") as f:
        res = json.load(f)
        for k, v in res.items():
            res[k] = [float(el) for el in v]

    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)

    bp = ax.boxplot(
        list(res.values()),
        labels=["Golden", "Ours", "Random"],
        patch_artist=True,
        showfliers=False,
    )
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.savefig("nsp_boxplot.png", dpi=300, bbox_inches="tight")
    plt.savefig("nsp_boxplot.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    # scoring_main()
    boxing_main()
