import argparse
import os
from functools import partial

import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from utils import get_logger, set_random_seed


def main(args):
    logger = get_logger()
    set_random_seed()
    device = torch.device("cuda")

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
    model.eval()
    model.to(device)

    for setname in ["valid", "train"]:
        random_fname = args.random_fname.format(setname)
        if setname == "train":
            args.output_fname = args.output_fname.replace("valid.txt", "train.txt")
        with open(random_fname, "r") as f:
            ls = [el.strip().split("|||") for el in f.readlines()]
            assert all([len(el) == 3 for el in ls])

        for line_idx, line in enumerate(tqdm(ls)):
            context, golden, rand = line
            if args.strategy == "prefix":
                tokenized_golden = tokenizer(
                    ["<bos>" + golden],
                    return_tensors="pt",
                    max_length=128,
                    truncation=True,
                    padding=False,
                )["input_ids"]
                prefix_count = int(args.prefix_ratio * (len(tokenized_golden[0]) - 1))
                if prefix_count <= 3:
                    generated = "[NONE]"
                    ls[line_idx].append(generated)
                    continue
                model_input = torch.tensor(tokenized_golden).to(device)[
                    :, :prefix_count
                ]
                output_min_len = len(model_input[0]) + 3
            elif args.strategy == "del_prev_turn":
                context_list = context.strip().split("[SEPT]")
                if len(context_list) == 1:
                    generated = "[NONE]"
                    ls[line_idx].append(generated)
                    continue
                context = context_list[:-1]
                model_input = "<bos> " + " ".join(context) + " <sep>"
                model_input = tokenizer(model_input)["input_ids"]
                if len(model_input) >= 128:
                    model_input = model_input[-100:]
                output_min_len = len(model_input) + 5
                model_input = torch.tensor([model_input]).to(device)

            partial_generate_func = partial(
                model.generate,
                input_ids=model_input,
                eos_token_id=tokenizer.eos_token_id,
                min_length=output_min_len,
                max_length=128,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=args.do_sample,
                no_repeat_ngram_size=3,
            )
            if args.decode == "topp":
                generated = partial_generate_func(top_p=args.topp)
            elif args.decode == "topk":
                generated = partial_generate_func(top_k=args.topk)
            elif args.decode == "beam":
                generated = partial_generate_func(num_beams=args.beam_size)
            else:
                raise ValueError
            assert len(generated) == 1
            # To consider the context length
            if not args.is_gpt_sent_level:
                generated = [generated[0][len(model_input[0]) :]]

            result = tokenizer.decode(
                generated[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            print("Context: ", context)
            print("Golden: ", golden)
            print("Generate: ", result, "\n\n")

            ls[line_idx].append(result)
        assert all([len(el) == 2 + args.neg_num for el in ls])
        with open(args.output_fname, "w") as f:
            ls = ["|||".join(el) for el in ls]
            f.write("\n".join(ls))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--strategy",
        type=str,
        default="normal",
        choices=["normal", "prefix", "del_prev_turn"],
    )
    parser.add_argument(
        "--decode",
        type=str,
        default="beam",
        choices=["beam", "topp", "topk", "pure"],
    )
    parser.add_argument("--prefix_ratio", type=float, default=0.4)
    parser.add_argument("--beam_size", default=10)
    parser.add_argument("--topp", default=0.9)
    parser.add_argument("--topk", default=100)

    parser.add_argument(
        "--neg_num", type=int, default=2, help="negative의 총합(random 포함)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./logs/GPT2-FT/models/",
        choices=["./logs/sentGPT/models/", "./logs/GPT2-FT/models/"],
    )
    parser.add_argument(
        "--random_fname",
        type=str,
        default="./data/negative/random_neg1_{}.txt",
    )
    parser.add_argument(
        "--output_fname",
        type=str,
        default="./data/negative/{}_neg{}_valid.txt",
    )
    args = parser.parse_args()
    args.do_sample = False if args.decode in ["greedy", "beam"] else True
    args.is_gpt_sent_level = True if "sentGPT" in args.model_path else False

    decode_param = (
        args.topp
        if args.decode == "topp"
        else args.topk
        if args.decode == "topk"
        else args.beam_size
        if args.decode == "beam"
        else None
    )
    args.exp_name = f"{args.strategy}-{args.decode}{decode_param}"
    args.output_fname = args.output_fname.format(args.exp_name, args.neg_num)

    """Argument Integrity Check"""
    if args.strategy in ["normal", "del_prev_turn"]:
        assert not args.is_gpt_sent_level
    else:
        assert args.is_gpt_sent_level

    assert not os.path.exists(args.output_fname.format("valid"))
    """End"""
    for k, v in vars(args).items():
        print(k, " -> ", v)
    input("RIGHT? >>")
    main(args)
