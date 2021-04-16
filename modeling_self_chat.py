"""
eval set의 각 턴에서 한번씩 더 이어서 만들어본 다음에, 뭐가 나오는지 알아보기 + 어떤 경향성이 있는지 보기.
"""
import json
import os
from typing import List, Tuple

import torch
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BertForNextSentencePrediction, BertTokenizer,
                          GPT2LMHeadModel, GPT2Tokenizer)

from datasets import TURN_TOKEN, EvalDataset, NSPDataset
from get_dataset import get_zhao_dataset
from utils import set_random_seed


class Config:
    gpt_model_path = "./logs/GPT2-FT/models/"
    bertrank_model_path = "./logs_wo_ttype/random_neg1/model/epoch-0.pth"


def main():
    """
    Phase 0. Basic Configuration
    """
    set_random_seed(42)
    config = Config()
    softmax = torch.nn.Softmax(dim=1)
    device = torch.device("cuda")
    evalset_name = "dd"
    USE_FT_GPT = True
    USE_DGPT = True
    assert USE_FT_GPT + USE_DGPT >= 1
    """
    Phase 1. Generative LMs loading
    """
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_tokenizer.add_special_tokens({"additional_special_tokens": [TURN_TOKEN]})
    bert_ranker = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")
    bert_ranker.resize_token_embeddings(len(bert_tokenizer))
    bert_ranker.load_state_dict(torch.load(config.bertrank_model_path))
    bert_ranker.eval()
    bert_ranker.to(device)

    if USE_FT_GPT:
        gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        gpt_tokenizer.add_special_tokens(
            {
                "bos_token": "<bos>",
                "sep_token": "<sep>",
                "eos_token": "<eos>",
                "pad_token": "<pad>",
            }
        )
        gpt = GPT2LMHeadModel.from_pretrained(config.gpt_model_path)
        gpt.eval()
        gpt.to(device)
    if USE_DGPT:
        dgpt_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        dgpt = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
        dgpt.eval()
        dgpt.to(device)

    """
    Phase 2. Evaluation Dataset Loading
    """
    # ctx: List of utterances(str)/ hyp and ref: utterance (str)
    raw_eval_dataset = get_zhao_dataset(evalset_name)

    """
    Phase 3. Generation with various setup
    """

    saver = []
    for idx, sample in enumerate(tqdm(raw_eval_dataset)):
        ctx: List[str] = sample["ctx"]
        hyp: str = sample["hyp"]
        ref: str = sample["ref"]

        def generate_and_eval(
            generate_strategy: str, evaluation_strategy: str, genmodel_name
        ) -> Tuple[str, float]:
            """
            Generation Context를 어떻게 할지 (c+h, h only)
            Evaluation을 어떻게 할지 (c+h->s, h->s)
            """
            assert generate_strategy in ["ch", "h"]
            assert evaluation_strategy in ["ch", "h"]
            assert genmodel_name in ["gpt", "dgpt"]
            assert isinstance(ctx, list) and all([isinstance(el, str) for el in ctx])
            assert isinstance(hyp, str)

            if genmodel_name == "dgpt":
                if generate_strategy == "ch":
                    generative_input = dgpt_tokenizer.encode(
                        dgpt_tokenizer.eos_token.join(ctx + [hyp])
                        + dgpt_tokenizer.eos_token,
                        return_tensors="pt",
                    ).to(device)
                else:
                    generative_input = dgpt_tokenizer.encode(
                        hyp + dgpt_tokenizer.eos_token,
                        return_tensors="pt",
                    ).to(device)
                generated = dgpt.generate(
                    generative_input,
                    max_length=500,
                    top_k=100,
                    pad_token_id=dgpt_tokenizer.eos_token_id,
                    num_beams=5,
                    do_sample=True,
                )[:, len(generative_input[0]) :]
                decoded = dgpt_tokenizer.decode(generated[0], skip_special_tokens=True)
            else:
                if generate_strategy == "ch":
                    generative_input = gpt_tokenizer.encode(
                        gpt_tokenizer.bos_token
                        + " ".join(ctx + [hyp])
                        + gpt_tokenizer.sep_token,
                        return_tensors="pt",
                    ).to(device)
                else:
                    generative_input = gpt_tokenizer.encode(
                        gpt_tokenizer.bos_token + hyp + gpt_tokenizer.sep_token,
                        return_tensors="pt",
                    ).to(device)
                generated = gpt.generate(
                    generative_input,
                    max_length=500,
                    top_k=100,
                    pad_token_id=gpt_tokenizer.eos_token_id,
                    eos_token_id=gpt_tokenizer.eos_token_id,
                    num_beams=5,
                    do_sample=True,
                )[:, len(generative_input[0]) :]
                decoded = gpt_tokenizer.decode(generated[0], skip_special_tokens=True)
            """
            Prediction
            """
            first_seq = (
                TURN_TOKEN.join(ctx + [hyp]) if evaluation_strategy == "ch" else hyp
            )
            ranker_encoded = bert_tokenizer(
                first_seq,
                text_pair=decoded,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                ids, masks = (
                    ranker_encoded["input_ids"].to(device),
                    ranker_encoded["attention_mask"].to(device),
                )
                prediction = float(
                    softmax(bert_ranker(ids, masks)[0]).cpu().numpy()[0][0]
                )
            return decoded, prediction

        """
        Original
        """
        hyp_ranker_encoded = bert_tokenizer(
            TURN_TOKEN.join(ctx),
            text_pair=hyp,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        hyp_ranker_encoded = bert_tokenizer(
            TURN_TOKEN.join(ctx),
            text_pair=hyp,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            ids, masks = (
                hyp_ranker_encoded["input_ids"].to(device),
                hyp_ranker_encoded["attention_mask"].to(device),
            )
            original_prediction = float(
                softmax(bert_ranker(ids, masks)[0]).cpu().numpy()[0][0]
            )
            """
            Variations
            """
            (
                sample["dgpt_ch_ch_decoded"],
                sample["dgpt_ch_ch_score"],
            ) = generate_and_eval(
                generate_strategy="ch", evaluation_strategy="ch", genmodel_name="dgpt"
            )
            sample["dgpt_ch_h_decoded"], sample["dgpt_ch_h_score"] = generate_and_eval(
                generate_strategy="ch", evaluation_strategy="h", genmodel_name="dgpt"
            )
            sample["dgpt_h_ch_decoded"], sample["dgpt_h_ch_score"] = generate_and_eval(
                generate_strategy="h", evaluation_strategy="ch", genmodel_name="dgpt"
            )
            sample["dgpt_h_h_decoded"], sample["dgpt_h_h_score"] = generate_and_eval(
                generate_strategy="h", evaluation_strategy="h", genmodel_name="dgpt"
            )
            sample["gpt_ch_ch_decoded"], sample["gpt_ch_ch_score"] = generate_and_eval(
                generate_strategy="ch", evaluation_strategy="ch", genmodel_name="gpt"
            )
            sample["gpt_ch_h_decoded"], sample["gpt_ch_h_score"] = generate_and_eval(
                generate_strategy="ch", evaluation_strategy="h", genmodel_name="gpt"
            )
            sample["gpt_h_ch_decoded"], sample["gpt_h_ch_score"] = generate_and_eval(
                generate_strategy="h", evaluation_strategy="ch", genmodel_name="gpt"
            )
            sample["gpt_h_h_decoded"], sample["gpt_h_h_score"] = generate_and_eval(
                generate_strategy="h", evaluation_strategy="h", genmodel_name="gpt"
            )
            sample["original_pred"] = original_prediction
        saver.append(sample)

    with open("saver_topk.jsonl", "w") as f:
        for line in saver:
            json.dump(line, f)
            f.write("\n")

    """
    Phase 4. Save and Visualization
    """


if __name__ == "__main__":
    main()
