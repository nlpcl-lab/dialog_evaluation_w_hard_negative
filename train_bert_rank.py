import argparse
import os

import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm import tqdm
from transformers import BertConfig, BertModel, BertTokenizer

from bert_rank_model import BertRankModel
from datasets import TURN_TOKEN, EvalDataset, NSPDataset
from get_dataset import get_dd_corpus, get_zhao_dataset
from trainer import Trainer
from utils import (dump_config, get_logger, load_model, save_model,
                   set_random_seed, write_summary)


def main(args):
    set_random_seed(42)
    logger = get_logger()

    dump_config(args)
    device = torch.device("cuda")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    special_tokens_dict = {"additional_special_tokens": [TURN_TOKEN]}
    tokenizer.add_special_tokens(special_tokens_dict)
    bert = BertModel.from_pretrained("bert-base-uncased")
    bert.resize_token_embeddings(len(tokenizer))
    bert_config = BertConfig.from_pretrained("bert-base-uncased")
    model = BertRankModel(bert, bert_config, args)
    model.to(device)

    dataset_for_correlation = EvalDataset(
        get_zhao_dataset("dd"), 128, tokenizer, rank_loss=True
    )

    train_dataset, valid_dataset = (
        NSPDataset(
            args.data_path.format(args.num_neg, "train"),
            128,
            tokenizer,
            num_neg=args.num_neg,
            rank_loss=True,
        ),
        NSPDataset(
            args.data_path.format(args.num_neg, "valid"),
            128,
            tokenizer,
            num_neg=args.num_neg,
            rank_loss=True,
        ),
    )
    trainloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=args.batch_size,
        drop_last=True,
    )
    validloader = DataLoader(valid_dataset, batch_size=args.batch_size, drop_last=True)

    trainer = Trainer(
        args,
        model,
        trainloader,
        validloader,
        logger,
        device,
        dataset_for_correlation,
        is_rank_loss=True,
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--exp_name",
        type=str,
        # "./data/negative/neg{}_{}_k1_maxchange0.5_minchange0.15_nspoveronly0.3.txt"
        default="k1_maxchange0.5_minchange0.15_nspoveronly0.3",
    )
    parser.add_argument("--num_neg", type=int, default=2)
    parser.add_argument("--log_path", type=str, default="rank_logs")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument(
        "--data_path",
        type=str,
        # "./data/negative/neg{}_{}_k5_maxchange0.5_nspoveronly0.3_scorediff0.01",  # "./data/negative/neg{}_{}_k1_maxchange1.0_nspover0.3_scorediff0.05.txt",  # "./data/negative/neg{}_{}_pred5_numtokenratio0.5_nspthreshold0.3_scorediff0.01.txt",  # "./data/negative/del_prev_turn-topk100_neg{}_{}.txt",  # "./data/negative/prefix-topk100_neg{}_{}.txt",
        # './data/negative/random_neg{}_{}.txt',
        default="./data/negative/neg{}_{}_k1_maxchange0.5_minchange0.15_nspoveronly0.3.txt",
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
