import argparse
import os

import torch
from tensorboardX import SummaryWriter
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm import tqdm
from transformers import BertConfig, BertForNextSentencePrediction, BertTokenizer

from datasets import TURN_TOKEN, GraphEvalDataset, GraphNSPDataset
from get_dataset import get_dd_corpus, get_zhao_dataset
from trainer import Trainer
from utils import dump_config, get_logger, load_model, save_model, set_random_seed, write_summary
from graph_preprocess import *
from graph_model import GRADE


def main(args):
    max_keyword_num = 10
    embedding_size = 50000

    set_random_seed(42)
    logger = get_logger()

    dump_config(args)
    device = torch.device("cpu")  # "cuda")
    c2i, i2c, r2i, i2r, w2v = load_resources()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    special_tokens_dict = {"additional_special_tokens": [TURN_TOKEN]}
    tokenizer.add_special_tokens(special_tokens_dict)

    model = GRADE(w2v, len(tokenizer))
    model.to(device)

    cpnet = load_cpnet()

    dataset_for_correlation = GraphEvalDataset(
        get_zhao_dataset("dd"), 128, tokenizer, cpnet, c2i, i2c, w2v, max_keyword_num
    )

    train_dataset, valid_dataset = (
        GraphNSPDataset(
            args.data_path.format("train"),  # (args.num_neg, "train"),
            128,
            tokenizer,
            cpnet,
            c2i,
            i2c,
            w2v,
            max_keyword_num,
        ),
        GraphNSPDataset(
            args.data_path.format("valid"),  # (args.num_neg, "valid"),
            128,
            tokenizer,
            cpnet,
            c2i,
            i2c,
            w2v,
            max_keyword_num,
        ),
    )

    trainloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=args.batch_size,
        drop_last=True,
    )
    validloader = DataLoader(valid_dataset, batch_size=args.batch_size, drop_last=True)
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
    )
    criteria = torch.nn.BCELoss()
    for step, batch in enumerate(tqdm(trainloader)):
        ids, _, masks, labels, keywords, adjs = [el.to(device) for el in batch]
        res = model(ids, masks, keywords, adjs)
        loss = criteria(res, labels)
        loss.backward
        optimizer.step()
        optimizer.zero_grad()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--exp_name",
        type=str,
        default="graph_naive",
    )
    parser.add_argument("--num_neg", type=int, default=2)
    parser.add_argument("--log_path", type=str, default="logs_wo_ttype")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/negative/random_neg1_{}.txt",
    )

    args = parser.parse_args()

    args.exp_path = os.path.join(args.log_path, args.exp_name)
    args.model_path = os.path.join(args.exp_path, "model")
    args.board_path = os.path.join(args.exp_path, "board")
    from pprint import pprint

    pprint(args)
    input(">> RIGHT?")
    os.makedirs(args.model_path, exist_ok=False)
    os.makedirs(args.board_path, exist_ok=False)
    main(args)
