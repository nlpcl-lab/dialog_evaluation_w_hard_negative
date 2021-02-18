import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from transformers import BertTokenizer, BertModel, BertConfig
from utils import (
    set_random_seed,
    dump_config,
    save_model,
    load_model,
    write_summary,
    get_correlation,
)
from get_dataset import get_dd_corpus, get_zhao_dataset
import argparse
import os
from torch.optim.adamw import AdamW
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
from numpy import linalg

TURN_TOKEN = "[SEPT]"


def make_annotated_dataset(raw_zhao_data, tokenizer):
    values = []
    for item in raw_zhao_data:
        ctx = "[SEPT]".join(item["ctx"])
        ref = item["ref"]
        hyp = item["hyp"]
        score = item["human_score"]
        one_item = []
        one_item.append(torch.tensor(tokenizer(ctx)["input_ids"][:128]))
        one_item.append(torch.tensor(tokenizer(ref)["input_ids"][:128]))
        one_item.append(torch.tensor(tokenizer(hyp)["input_ids"][:128]))
        one_item.append(score)
        values.append(one_item)
    return values


def main(args):
    set_random_seed(42)
    device = torch.device("cuda")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    special_tokens_dict = {"additional_special_tokens": ["[SEPT]"]}
    tokenizer.add_special_tokens(special_tokens_dict)

    context_encoder = load_model(
        BertModel(BertConfig.from_pretrained("bert-base-uncased")),
        args.context_model_path,
        5,
        len(tokenizer),
    )
    response_encoder = load_model(
        BertModel(BertConfig.from_pretrained("bert-base-uncased")),
        args.response_model_path,
        5,
        len(tokenizer),
    )
    context_encoder.to(device)
    response_encoder.to(device)

    test_data = make_annotated_dataset(get_zhao_dataset("dd"), tokenizer)
    # test_data = make_annotated_dataset(get_zhao_dataset("persona"), tokenizer)

    humanscore, refscore, ctxscore, meanscore = [], [], [], []
    refdistance, ctxdistance = [], []
    for item_idx, item in enumerate(test_data):
        if item_idx % 200 == 0:
            print(item_idx, len(test_data))
        ctx, ref, hyp, score = item
        with torch.no_grad():
            ctx_encoded = context_encoder(
                ctx.unsqueeze(0).to(device), return_dict=True
            )["pooler_output"]
            hyp_encoded = response_encoder(
                hyp.unsqueeze(0).to(device), return_dict=True
            )["pooler_output"]
            ref_encoded = response_encoder(
                ref.unsqueeze(0).to(device), return_dict=True
            )["pooler_output"]

            ctx_hyp_score = (
                torch.matmul(ctx_encoded, hyp_encoded.T)[0][0]
                .cpu()
                .detach()
                .numpy()
            )
            ref_hyp_score = (
                torch.matmul(ref_encoded, hyp_encoded.T)[0][0]
                .cpu()
                .detach()
                .numpy()
            )
        refdistance.append(
            linalg.norm(
                (ref_encoded - hyp_encoded).cpu().detach().numpy(), ord=2
            )
        )
        ctxdistance.append(
            linalg.norm(
                (ctx_encoded - hyp_encoded).cpu().detach().numpy(), ord=2
            )
        )
        humanscore.append(score)
        refscore.append(float(ref_hyp_score))
        ctxscore.append(float(ctx_hyp_score))
        meanscore.append(float(ref_hyp_score + ctx_hyp_score) / 2)

    refscore = get_correlation(humanscore, refscore)
    ctxscore = get_correlation(humanscore, ctxscore)
    meanscore = get_correlation(humanscore, meanscore)
    print("reference: ", refscore)
    print("context: ", ctxscore)
    print("ensemble: ", meanscore)
    print("ref_l2: ", get_correlation(humanscore, refdistance))
    print("ctx_l2: ", get_correlation(humanscore, ctxdistance))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--exp_name", type=str, default="test")
    parser.add_argument("--log_path", type=str, default="logs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epoch", type=int, default=10)

    args = parser.parse_args()

    args.exp_path = os.path.join(args.log_path, args.exp_name)
    args.context_model_path = os.path.join(args.exp_path, "context_model")
    args.response_model_path = os.path.join(args.exp_path, "response_model")
    args.board_path = os.path.join(args.exp_path, "board")
    os.makedirs(args.context_model_path, exist_ok=True)
    os.makedirs(args.response_model_path, exist_ok=True)
    os.makedirs(args.board_path, exist_ok=True)
    main(args)
