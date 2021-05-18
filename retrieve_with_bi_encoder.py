import argparse
import os

import numpy as np
import torch
from numpy import linalg
from tensorboardX import SummaryWriter
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm import tqdm
from transformers import BertConfig, BertModel, BertTokenizer

from get_dataset import get_dd_corpus, get_zhao_dataset, get_persona_corpus
from utils import (
    dump_config,
    get_correlation,
    draw_scatter_plot,
    load_model,
    save_model,
    softmax_2d,
    set_random_seed,
    write_summary,
)

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


from train_bi_encoder import BiEncoder


def main(args):
    set_random_seed(42)
    device = torch.device("cuda")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    special_tokens_dict = {"additional_special_tokens": ["[SEPT]"]}
    tokenizer.add_special_tokens(special_tokens_dict)
    contextbert, responsebert = (
        BertModel.from_pretrained("bert-base-uncased"),
        BertModel.from_pretrained("bert-base-uncased"),
    )
    contextbert.resize_token_embeddings(len(tokenizer))
    responsebert.resize_token_embeddings(len(tokenizer))
    context_encoder = BiEncoder(768, contextbert)
    response_encoder = BiEncoder(768, responsebert)
    context_encoder.load_state_dict(torch.load(args.context_model_path + "/epoch-{}.pth".format(0)))
    response_encoder.load_state_dict(
        torch.load(args.response_model_path + "/epoch-{}.pth".format(0))
    )
    context_encoder.eval()
    response_encoder.eval()
    context_encoder.to(device)
    response_encoder.to(device)

    
    if args.eval_set == 'dd':
        raw_dd_train = get_dd_corpus("train")
        test_data = make_annotated_dataset(get_zhao_dataset("dd"), tokenizer)
    else:
        raw_dd_train = get_persona_corpus("train")
        test_data = make_annotated_dataset(get_zhao_dataset("persona"), tokenizer)

    if args.eval_protocal == "direct":
        humanscore, ctxscore = [[], []]
        for item_idx, item in enumerate(tqdm(test_data)):
            ctx, ref, hyp, score = item
            with torch.no_grad():
                ctx_encoded = context_encoder(
                    ctx.unsqueeze(0).to(device),
                    torch.tensor([1 for _ in range(len(ctx))]).unsqueeze(0).to(device),
                )
                hyp_encoded = response_encoder(
                    hyp.unsqueeze(0).to(device),
                    torch.tensor([1 for _ in range(len(hyp))]).unsqueeze(0).to(device),
                )
                ctx_hyp_score = (
                    torch.matmul(ctx_encoded, hyp_encoded.T)[0][0].cpu().detach().numpy()
                )
            humanscore.append(score)
            ctxscore.append(float(ctx_hyp_score))

        ctxscore = get_correlation(humanscore, ctxscore)
        print("context: ", ctxscore)
    elif args.eval_protocal == "rank":
        all_uttrs = [uttr for conv in raw_dd_train for uttr in conv]
        cached_vecs = caching_response(
            all_uttrs, args.response_cache_fname, response_encoder, tokenizer, device
        )
        humanscore = [el[-1] for el in test_data]
        retrived_score, hyp_score = retrieve_candidate(
            args.retrieval_cache_fname,
            args.hyp_cache_fname,
            context_encoder,
            response_encoder,
            tokenizer,
            device,
            test_data,
            cached_vecs,
        )
        expanded_hyp_score = np.expand_dims(hyp_score, 1)
        concated_score = np.concatenate((expanded_hyp_score, retrived_score), 1)
        argsorted = np.argsort(-concated_score)
        rank = np.where(argsorted == 0)[1]

        print("Naive: {}".format(get_correlation(humanscore, hyp_score)))
        print("rank: {}".format(get_correlation(humanscore, rank)))
        draw_scatter_plot(title="naive", x=humanscore, y=hyp_score, fname="./scatter/naive.png")
        draw_scatter_plot(title="rank_all", x=humanscore, y=rank, fname="./scatter/rank_all.png")
        
        


def caching_response(all_uttrs, fname, encoder, tokenizer, device):
    if os.path.exists(fname):
        with open(fname, "rb") as f:
            return np.load(f)
    cache = []
    for uttr in tqdm(all_uttrs):
        encoded = tokenizer(uttr, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            encoded = encoder(encoded).cpu().detach().numpy()[0]
            cache.append(encoded)
    cache = np.array(cache)
    with open(fname, "wb") as f:
        np.save(f, cache)
    return cache


def retrieve_candidate(
    retrieval_cache_fname,
    hyp_cache_fname,
    context_encoder,
    response_encoder,
    tokenizer,
    device,
    test_data,
    cached_vecs,
):
    if os.path.exists(retrieval_cache_fname) and os.path.exists(hyp_cache_fname):
        with open(retrieval_cache_fname, "rb") as f:
            a = np.load(f)
        with open(hyp_cache_fname, "rb") as f:
            b = np.load(f)
            return a, b
    context_encoded_list = []
    hyp_encoded_list = []
    for item_idx, item in enumerate(tqdm(test_data)):
        ctx, ref, hyp, score = item
        with torch.no_grad():
            ctx_encoded = context_encoder(ctx.unsqueeze(0).to(device)).cpu().detach().numpy()[0]
            context_encoded_list.append(ctx_encoded)
            hyp_encoded = response_encoder(hyp.unsqueeze(0).to(device)).cpu().detach().numpy()[0]
            hyp_encoded_list.append(hyp_encoded)

    context_encoded_list = np.array(context_encoded_list)
    hyp_encoded_list = np.array(hyp_encoded_list)

    retrieval_score = np.matmul(context_encoded_list, cached_vecs.T)
    hyp_score = np.sum(context_encoded_list * hyp_encoded_list, 1)

    with open(retrieval_cache_fname, "wb") as f:
        np.save(f, retrieval_score)
    with open(hyp_cache_fname, "wb") as f:
        np.save(f, hyp_score)
    return retrieval_score, hyp_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--exp_name", type=str, default="bi_encoder")
    parser.add_argument("--log_path", type=str, default="logs")
    parser.add_argument("--eval_set", type=str, default="dd", choices=["dd", "persona"])
    parser.add_argument("--eval_protocal", default="rank", choices=["direct", "rank"])  
    parser.add_argument("--cache_dir", type=str, default="./data/cache")

    args = parser.parse_args()
    if args.eval_set =='persona':
        args.exp_name += '_persona'

    args.exp_path = os.path.join(args.log_path, args.exp_name)
    args.context_model_path = os.path.join(args.exp_path, "context_model")
    args.response_model_path = os.path.join(args.exp_path, "response_model")
    args.board_path = os.path.join(args.exp_path, "board")
    args.response_cache_fname = os.path.join(args.cache_dir, "{}-train-bi64-epoch0.npy").format(
        args.eval_set
    )
    args.retrieval_cache_fname = os.path.join(
        args.cache_dir, "{}-response-bi64-epoch0-1000.json".format(args.eval_set)
    )
    args.hyp_cache_fname = os.path.join(
        args.cache_dir, "{}-hyp-bi64-epoch0-1000.json".format(args.eval_set)
    )
    os.makedirs(os.path.dirname(args.response_cache_fname), exist_ok=True)
    from pprint import pprint
    pprint(args)
    main(args)
