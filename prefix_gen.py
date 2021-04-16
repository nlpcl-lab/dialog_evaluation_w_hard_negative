import argparse
import os
from functools import partial

import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel,
                          GPT2Tokenizer)

from utils import get_logger, set_random_seed


def main(args):
    logger = get_logger()
    set_random_seed()
    device = torch.device("cuda")
    if args.model == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.add_special_tokens(
            {
                "bos_token": "<bos>",
                "sep_token": "<sep>",
                "eos_token": "<eos>",
                "pad_token": "<pad>",
            }
        )
        model = GPT2LMHeadModel.from_pretrained(
            args.model_path, pad_token_id=tokenizer.eos_token_id
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

    model.eval()
    model.to(device)

    for setname in ["valid", "train"]:
        random_fname = args.random_fname.format(setname)
        if setname == "train":
            args.output_fname = args.output_fname.replace("valid", "train")
        with open(random_fname, "r") as f:
            ls = [el.strip().split("|||") for el in f.readlines()]
            assert all([len(el) == 3 for el in ls])

        for line_idx, line in enumerate(tqdm(ls)):
            context, golden, rand = line
            if args.model == "gpt2":
                tokenized_golden = tokenizer(
                    ["<bos>" + golden],
                    return_tensors="pt",
                    max_length=128,
                    truncation=True,
                    padding=False,
                )["input_ids"]
            else:
                tokenized_golden = tokenizer.encode(golden, return_tensors="pt")

            prefix_count = int(args.prefix_ratio * (len(tokenized_golden[0]) - 1))
            if prefix_count <= 3:
                generated = ["[NONE]" for _ in range(args.neg_num - 1)]
                ls[line_idx].extend(generated)
                continue
            model_input = torch.tensor(tokenized_golden).to(device)[:, :prefix_count]
            output_min_len = len(model_input[0]) + 3

            partial_generate_func = partial(
                model.generate,
                input_ids=model_input,
                eos_token_id=tokenizer.eos_token_id,
                min_length=output_min_len,
                max_length=128,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=args.do_sample,
                no_repeat_ngram_size=3,
                num_return_sequences=args.neg_num - 1,
                length_penalty=args.length_penalty,
            )
            if args.decode == "topp":
                generated = partial_generate_func(top_p=args.topp)
            elif args.decode == "topk":
                generated = partial_generate_func(top_k=args.topk)
            elif args.decode == "beam":
                generated = partial_generate_func(num_beams=args.beam_size)
            else:
                raise ValueError
            assert len(generated) == args.neg_num - 1
            # To consider the context length

            result = [
                tokenizer.decode(
                    el,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                for el in generated
            ]
            for tmp_idx, tmp_gen in enumerate(result):
                while "\n" in tmp_gen:
                    tmp_gen = tmp_gen.replace("\n", " ")
                result[tmp_idx] = tmp_gen

            print("Context: ", context)
            print("Golden: ", golden)
            print("Generate: ", result, "\n\n")

            ls[line_idx].extend(result)

        assert all([len(el) == 2 + args.neg_num for el in ls])
        with open(args.output_fname, "w") as f:
            ls = ["|||".join(el) for el in ls]
            f.write("\n".join(ls))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--model", type=str, default="gpt2", choices=["gpt2", "dialoggpt"]
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="prefix",
        choices=["prefix"],
    )
    parser.add_argument(
        "--decode",
        type=str,
        default="topk",
        choices=["beam", "topp", "topk", "pure"],
    )
    parser.add_argument("--prefix_ratio", type=float, default=0.5)
    parser.add_argument("--beam_size", default=10)
    parser.add_argument("--topp", default=0.9)
    parser.add_argument("--topk", default=50, type=int)

    parser.add_argument(
        "--neg_num", type=int, default=2, help="negative의 총합(random 포함)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./logs/sentGPT/models/",
        choices=["./logs/sentGPT/models/"],
    )
    parser.add_argument(
        "--random_fname",
        type=str,
        default="./data/negative/random_neg1_{}.txt",
    )
    parser.add_argument(
        "--output_fname",
        type=str,
        default="./data/negative/neg{}_valid_{}.txt",
    )
    parser.add_argument(
        "--length_penalty",
        type=float,
        default=0.9,
    )
    args = parser.parse_args()
    args.do_sample = False if args.decode in ["greedy", "beam"] else True
    assert isinstance(args.neg_num, int)
    decode_param = (
        args.topp
        if args.decode == "topp"
        else args.topk
        if args.decode == "topk"
        else args.beam_size
        if args.decode == "beam"
        else None
    )
    args.exp_name = f"{args.strategy}{args.prefix_ratio}-{args.decode}{decode_param}"
    if args.length_penalty != 1.0:
        args.exp_name += "-lenpenalty{}".format(args.length_penalty)
    if args.model != "gpt2":
        args.exp_name = "dialoggpt-" + args.exp_name
    args.output_fname = args.output_fname.format(args.neg_num, args.exp_name)
    print(args.output_fname)
    assert not os.path.exists(args.output_fname)
    """End"""
    for k, v in vars(args).items():
        print(k, " -> ", v)
    input("RIGHT? >>")
    main(args)
