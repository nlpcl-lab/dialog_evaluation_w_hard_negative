"""
Experimental code to attack the golden resposne with BERT-retrieval model
"""
import argparse
import os
import numpy as np
from functools import partial
from tqdm import tqdm
import json
import torch
from tqdm import tqdm
from transformers import (
    BertConfig,
    BertForNextSentencePrediction,
    BertTokenizer,
    BertForMaskedLM,
)
from sklearn.metrics.pairwise import cosine_similarity

from datasets import TURN_TOKEN, EvalDataset, NSPDataset
from utils import get_logger, read_raw_file, set_random_seed, get_usencoder

NUM_TOPK_PREDICTION = 5
MIN_CHANGE_RATIO = 0.1
MAX_CHANGE_RATIO = 0.3
STOP_NSP_THRESHOLD = 0.4
# SCORE_DIFF = 0.005
SCORE_DIFF = -100
USE_USE = False


def attack():
    use = get_usencoder()
    logger = get_logger()
    set_random_seed()
    device = torch.device("cuda")

    """
    Load Model
    """
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    special_tokens_dict = {"additional_special_tokens": ["[SEPT]"]}
    tokenizer.add_special_tokens(special_tokens_dict)
    modelconfig = BertConfig.from_pretrained("bert-base-uncased")
    model = BertForNextSentencePrediction(modelconfig)
    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load("./logs/rand_neg1/model/epoch-0.pth"))
    model.eval()
    model.to(device)

    bert_mlm_config = BertConfig.from_pretrained("bert-base-uncased")
    bert_mlm = BertForMaskedLM(bert_mlm_config)
    bert_mlm.load_state_dict(
        torch.load("./logs/ft_bert/model/epoch-0.pth"), strict=False
    )
    bert_mlm.to(device)

    """
    Load Dataset
    """
    train_raw, valid_raw = (
        read_raw_file("./data/negative/random_neg1_train.txt"),
        read_raw_file("./data/negative/random_neg1_valid.txt"),
    )
    os.makedirs("attack", exist_ok=True)
    softmax = torch.nn.Softmax(dim=0)

    for dataset_index, dataset in enumerate([valid_raw, train_raw]):
        load_fname = (
            "vurnerable/train" if dataset_index == 1 else "vurnerable/valid"
        )
        with open(load_fname, "r") as f:
            score_data = [json.loads(el) for el in f.readlines()]
        assert len(score_data) == len(dataset)

        result = []
        counter = {'made': 0, 'length': 0,
                   'low_original': 0, 'threshold': 0, 'min_change': 0, 'error': 0}
        for idx, item in enumerate(tqdm(score_data)):
            for k, v in counter.items():
                print(k, v)
            print()
            if "vurnerable_nsp" not in item:
                counter['error'] += 1
                result.append("[NONE]")
                continue
            (
                context,
                response,
                tokenized_response,
                original_score,
                vurnerable_score,
            ) = (
                item["context"],
                item["response"],
                item["tokenized_response"],
                item["original_nsp"],
                item["vurnerable_nsp"],
            )
            if original_score < 0.5:
                counter['low_original'] += 1
                result.append("[NONE]")
                continue

            original_encoded = tokenizer(
                context,
                text_pair=response,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            original_emb = use([response])[0]

            response_length = int(sum(original_encoded["token_type_ids"][0]))
            assert response_length == len(tokenized_response) + 1
            context_length = (
                int(sum(original_encoded["attention_mask"][0]))
                - response_length
            )
            if response_length <= 6:
                counter['length'] += 1
                result.append("[NONE]")
                continue

            original_score = get_nsp_score(original_encoded, model, device)
            # Double check the consistency of the score
            if original_score < 0.5:
                counter['low_original'] += 1
                result.append("[NONE]")
                continue

            input_ids = original_encoded["input_ids"].clone().detach()
            assert len(vurnerable_score) == len(tokenized_response)
            score_diff_list = [el - original_score for el in vurnerable_score]
            sorted_score_diff_list = sorted(
                range(len(score_diff_list)), key=lambda k: score_diff_list[k]
            )
            attacked_response = tokenized_response[:]
            lowest_token, lowest_score = None, 100

            changed_counter = 0
            for changed_num, token_index in enumerate(sorted_score_diff_list):
                if score_diff_list[token_index] > -SCORE_DIFF:
                    break
                if (
                    changed_counter / len(sorted_score_diff_list)
                    > MAX_CHANGE_RATIO
                ):
                    break

                original_token_id = tokenizer._convert_token_to_id(
                    attacked_response[token_index]
                )
                attacked_response[token_index] = tokenizer.mask_token
                model_input = torch.tensor(
                    [
                        tokenizer.convert_tokens_to_ids(
                            [tokenizer.cls_token]
                            + attacked_response
                            + [tokenizer.sep_token]
                        )
                    ]
                ).to(device)
                with torch.no_grad():
                    output = bert_mlm(model_input)[0]

                score_in_response = (
                    softmax(output[0][token_index + 1]).cpu().detach()
                )
                score_in_response[original_token_id] = 0.0
                sorted_idx = torch.argsort(
                    score_in_response, descending=True
                )[:NUM_TOPK_PREDICTION]

                if not USE_USE:
                    lowest_token, lowest_score = None, 100
                    for likely_token in sorted_idx:
                        copied_input_ids = input_ids.clone().detach()
                        copied_input_ids[0, context_length +
                                         token_index] = likely_token
                        original_encoded["input_ids"] = copied_input_ids
                        nsp_score = get_nsp_score(
                            original_encoded, model, device)
                        if lowest_score > nsp_score:
                            lowest_score = nsp_score
                            lowest_token = likely_token
                else:
                    unsimilar_token, unsimilar_score = None, 100
                    for likely_token in sorted_idx:
                        tmp_attacked_response = attacked_response[:]
                        tmp_attacked_response[token_index] = tokenizer.convert_ids_to_tokens([
                                                                                             likely_token])[0]
                        attacked_emb = use(
                            [' '.join(tmp_attacked_response).replace(" ##", "")])[0]
                        cossim = cosine_similarity(
                            [original_emb], [attacked_emb])[0][0]
                        if cossim < unsimilar_score:
                            unsimilar_score = cossim
                            unsimilar_token = likely_token

                    copied_input_ids = input_ids.clone().detach()
                    copied_input_ids[0, context_length +
                                     token_index] = unsimilar_token
                    original_encoded["input_ids"] = copied_input_ids
                    lowest_score = get_nsp_score(
                        original_encoded, model, device)
                    lowest_token = unsimilar_token

                # 원래꺼 안건드렸는지 확인
                assert input_ids[0, context_length +
                                 token_index] == original_token_id
                if lowest_score > original_score:
                    continue
                '''
                Change
                '''
                input_ids[0, context_length + token_index] = lowest_token
                attacked_response[
                    token_index
                ] = tokenizer.convert_ids_to_tokens([lowest_token])[0]
                changed_counter += 1

                '''
                BREAK CONDITION
                '''
                if lowest_score < STOP_NSP_THRESHOLD:
                    if changed_counter / len(sorted_score_diff_list) > MIN_CHANGE_RATIO:
                        #print(changed_counter, len(sorted_score_diff_list))
                        break
            # if lowest_score >= STOP_NSP_THRESHOLD:
            #     counter['threshold'] += 1
            #     result.append("[NONE]")
            #     continue
            if changed_counter / len(sorted_score_diff_list) <= MIN_CHANGE_RATIO:
                counter['min_change'] += 1
                result.append("[NONE]")
                continue
            counter['made'] += 1
            result.append(" ".join(attacked_response).replace(" ##", ""))
            # print(tokenized_response)
            # print(attacked_response)
            # print()
            assert "##" not in result

        assert len(dataset) == len(result)

        # _nspoveronly{STOP_NSP_THRESHOLD}.txt"
        fname_suffix = f"_k{NUM_TOPK_PREDICTION}_maxchange{MAX_CHANGE_RATIO}_minchange{MIN_CHANGE_RATIO}_nonspcutoff.txt"
        if USE_USE:
            fname_suffix = fname_suffix.replace(".txt", "_usesort.txt")
        with open(
            "attack/"
            + "neg2_"
            + ["valid", "train"][dataset_index]
            + fname_suffix,
            "w",
        ) as f:
            for line_idx, line in enumerate(result):
                f.write("|||".join(dataset[line_idx] + [result[line_idx]]))
                if line_idx != len(result) - 1:
                    f.write("\n")


def scoring():
    logger = get_logger()
    set_random_seed()
    device = torch.device("cuda")

    """
    Load Model
    """
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    special_tokens_dict = {"additional_special_tokens": ["[SEPT]"]}
    tokenizer.add_special_tokens(special_tokens_dict)
    modelconfig = BertConfig.from_pretrained("bert-base-uncased")
    model = BertForNextSentencePrediction(modelconfig)
    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load("./logs/rand_neg1/model/epoch-0.pth"))
    model.eval()
    model.to(device)

    """
    Load Dataset
    """
    train_raw, valid_raw = (
        read_raw_file("./data/negative/random_neg1_train.txt"),
        read_raw_file("./data/negative/random_neg1_valid.txt"),
    )

    os.makedirs("vurnerable", exist_ok=True)
    for dataset_index, dataset in enumerate([valid_raw, train_raw]):
        result = []
        for conv_idx, conversation in enumerate(tqdm(dataset)):
            context, response, _ = conversation
            original_encoded = tokenizer(
                context,
                text_pair=response,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            tokenized_response = tokenizer.tokenize(response)
            original_score = get_nsp_score(original_encoded, model, device)

            """
            EXP1: response 한 토큰씩 masking해서 NSP Score 변화 보기
            """
            response_length = int(sum(original_encoded["token_type_ids"][0]))
            try:
                assert response_length == len(tokenized_response) + 1
            except:
                assert int(sum(original_encoded["attention_mask"][0])) == 128
                item = {
                    "context": context,
                    "response": response,
                    "tokenized_response": tokenized_response,
                    "original_nsp": original_score,
                }
                result.append(item)
                continue

            context_length = (
                int(sum(original_encoded["attention_mask"][0]))
                - response_length
            )
            score_per_position = []
            for response_token_index in range(
                response_length - 1
            ):  # To avoid [SEP] masking
                mask_position = response_token_index + context_length
                attacked_ids = original_encoded["input_ids"][0]
                attacked_ids = np.concatenate(
                    [
                        attacked_ids[:mask_position].numpy(),
                        attacked_ids[mask_position + 1:].numpy(),
                        [tokenizer.pad_token_id],
                    ]
                ).tolist()
                attacked_input = {
                    "input_ids": torch.tensor([attacked_ids]),
                    "token_type_ids": torch.tensor(
                        [
                            [0 for _ in range(context_length)]
                            + [1 for _ in range(response_length - 1)]
                            + [
                                0
                                for _ in range(
                                    128 - context_length - response_length + 1
                                )
                            ]
                        ]
                    ),
                    "attention_mask": torch.tensor(
                        [
                            [
                                1
                                for _ in range(
                                    context_length + response_length - 1
                                )
                            ]
                            + [
                                0
                                for i in range(
                                    128 - context_length - response_length + 1
                                )
                            ]
                        ]
                    ),
                }
                attacked_score = get_nsp_score(
                    attacked_input,
                    model,
                    device,
                )
                score_per_position.append(attacked_score)
            assert len(score_per_position) == len(tokenized_response)
            item = {
                "context": context,
                "response": response,
                "tokenized_response": tokenized_response,
                "original_nsp": original_score,
                "vurnerable_nsp": score_per_position,
            }
            result.append(item)
        fname = "valid" if dataset_index == 0 else "train"
        with open("vurnerable/" + fname, "w") as f:
            for el in result:
                json.dump(el, f)
                f.write("\n")


def get_nsp_score(sample, model, device) -> float:
    """
    input_ids~token_type_ids는 (1, max_seq_len)의 shape을 가짐.
    """
    input_ids, attention_masks, token_type_ids = (
        sample["input_ids"],
        sample["attention_mask"],
        sample["token_type_ids"],
    )
    softmax = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        output = (
            softmax(
                model(
                    input_ids.to(device),
                    attention_masks.to(device),
                    token_type_ids.to(device),
                )[0]
            )
            .cpu()
            .numpy()[0][0]
        )
    return float(output)


if __name__ == "__main__":
    # scoring()
    attack()
