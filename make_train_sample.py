"""
Context ||| Golden ||| Negative(random)이 있는 txt 파일을 만듬.
"""
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from transformers import BertTokenizer, BertModel, BertConfig
from utils import (
    set_random_seed,
    dump_config,
    save_model,
    load_model,
    write_summary,
)
from get_dataset import get_dd_corpus
import argparse
import os
from torch.optim.adamw import AdamW
from tensorboardX import SummaryWriter
from tqdm import tqdm
import random

TURN_TOKEN = "[SEPT]"


class NSPDataset(Dataset):
    def __init__(self, raw_data, max_seq_len: int, tokenizer):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.feature = self._make_feature(raw_data)

    def __len__(self):
        return len(self.feature[0])

    def __getitem__(self, idx):
        return tuple([el[idx] for el in self.feature])

    def _make_feature(self, raw_data):
        ctx_inp_ids, ctx_masks, res_inp_ids, res_masks = [
            [] for _ in range(4)
        ]

        for item_idx, item in enumerate(raw_data):
            if item_idx % 100 == 0:
                print(f"{item_idx}/{len(raw_data)}")
            assert isinstance(item, list) and all(
                [isinstance(el, str) for el in item]
            )

            if len(item) > 6:
                item = item[:6]

            for uttr_idx in range(len(item) - 1):
                uttrs = item[: uttr_idx + 2]
                assert len(uttrs) <= 6

                context, response = TURN_TOKEN.join(uttrs[:-1]), uttrs[-1]

                context = self.tokenizer(
                    context,
                    max_length=self.max_seq_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                response = self.tokenizer(
                    response,
                    max_length=self.max_seq_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                ctx_inp_ids.extend(context["input_ids"])
                ctx_masks.extend(context["attention_mask"])
                res_inp_ids.extend(response["input_ids"])
                res_masks.extend(response["attention_mask"])

        return (
            torch.stack(ctx_inp_ids),
            torch.stack(ctx_masks),
            torch.stack(res_inp_ids),
            torch.stack(res_masks),
        )


def make_annotated_dataset(raw_zhao_data, tokenizer):
    ctx_ids, ref_ids, hyp_ids, human_score = [], [], [], []
    for item in raw_zhao_data:
        ctx = "[SEPT]".join(item["ctx"])
        ref = item["ref"]
        hyp = item["hyp"]
        score = item["human_score"]
        ctx_ids.append(torch.tensor(tokenizer(ctx)["input_ids"][:128]))
        ref_ids.append(torch.tensor(tokenizer(ref)["input_ids"][:128]))
        hyp_ids.append(torch.tensor(tokenizer(hyp)["input_ids"][:128]))
        human_score.append(score)
    return ctx_ids, ref_ids, hyp_ids, human_score


def main():
    set_random_seed(42)
    NEGNUM = 2
    raw_dd_train, raw_dd_valid = get_dd_corpus("train"), get_dd_corpus(
        "validation"
    )

    for idx, dataset in enumerate([raw_dd_valid, raw_dd_train]):
        setname = "valid" if idx == 0 else "train"
        fname = "./data/negative/random_neg{}_{}.txt".format(NEGNUM, setname)
        # assert not os.path.exists(fname)
        context_list, response_list = [], []
        for item in dataset:
            for cont_beg_idx in range(len(item) - 1):
                uttrs = item[: cont_beg_idx + 1]
                if len(uttrs) > 5:
                    uttrs = uttrs[-5:]
                context = TURN_TOKEN.join(uttrs)
                response = item[cont_beg_idx + 1]

                context_list.append(context)
                response_list.append(response)
        assert len(context_list) == len(response_list)
        negative_list = []
        if NEGNUM == 2:
            negative_list2 = []

        for item_idx, response in enumerate(response_list):
            while True:
                selected = random.choice(response_list)
                assert isinstance(selected, str)
                set1, set2 = set(selected.split()), set(response.split())
                cov_score = (
                    len(set1.intersection(set2)) / len(set1)
                    + len(set1.intersection(set2)) / len(set2)
                ) / 2

                if selected != response and cov_score < 0.8:
                    break
            negative_list.append(selected)
            if NEGNUM == 2:
                while True:
                    selected = random.choice(response_list)
                    assert isinstance(selected, str)
                    set1, set2 = set(selected.split()), set(response.split())
                    cov_score = (
                        len(set1.intersection(set2)) / len(set1)
                        + len(set1.intersection(set2)) / len(set2)
                    ) / 2

                    if selected != response and cov_score < 0.8:
                        break
                negative_list2.append(selected)

        assert (
            len(context_list)
            == len(response_list)
            == len(negative_list)
            == len(negative_list2)
        )
        with open(fname, "w") as f:
            for line_idx in range(len(context_list)):
                if NEGNUM == 1:
                    f.write(
                        "|||".join(
                            [
                                context_list[line_idx],
                                response_list[line_idx],
                                negative_list[line_idx],
                            ]
                        )
                    )
                elif NEGNUM == 2:
                    f.write(
                        "|||".join(
                            [
                                context_list[line_idx],
                                response_list[line_idx],
                                negative_list[line_idx],
                                negative_list2[line_idx],
                            ]
                        )
                    )
                else:
                    raise ValueError
                if line_idx != len(context_list) - 1:
                    f.write("\n")


if __name__ == "__main__":

    main()
