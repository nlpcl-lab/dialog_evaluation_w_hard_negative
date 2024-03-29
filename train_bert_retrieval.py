import argparse
import os

import torch
from tensorboardX import SummaryWriter
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm import tqdm
from transformers import BertConfig, BertForNextSentencePrediction, BertTokenizer

from datasets import TURN_TOKEN, EvalDataset, NSPDataset
from get_dataset import get_dd_corpus, get_zhao_dataset
from trainer import Trainer
from utils import dump_config, get_logger, load_model, save_model, set_random_seed, write_summary


def main(args):
    set_random_seed(42)
    logger = get_logger()

    dump_config(args)
    device = torch.device("cuda")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    special_tokens_dict = {"additional_special_tokens": [TURN_TOKEN]}
    tokenizer.add_special_tokens(special_tokens_dict)

    model = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    dataset_for_correlation = EvalDataset(
        get_zhao_dataset("dd"),
        128,
        tokenizer,
    )

    train_dataset, valid_dataset = (
        NSPDataset(
            args.data_path.format("train"),  # (args.num_neg, "train"),
            128,
            tokenizer,
            num_neg=args.num_neg,
        ),
        NSPDataset(
            args.data_path.format("valid"),  # (args.num_neg, "valid"),
            128,
            tokenizer,
            num_neg=args.num_neg,
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
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--exp_name",
        type=str,
        default="bart-mask0.5_topp0.95",  # "k5_maxchange0.4_minchange0.1_NSPCUT0.4",  # "random_neg1",
    )
    parser.add_argument("--num_neg", type=int, default=2)
    parser.add_argument("--log_path", type=str, default="logs_wo_ttype")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/negative/bart_mask0.5_topp0.95_{}_usesim0.3-0.9.txt",
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
