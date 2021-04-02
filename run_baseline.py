import argparse
import copy
import json
import os
import pickle
from typing import Dict, List

import numpy as np
import torch
from bert_score import score


from nlgeval import NLGEval
from numpy import dot
from numpy.linalg import norm
from scipy.stats import pearsonr, spearmanr
from tensorboardX import SummaryWriter
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm import tqdm
from transformers import (
    BertConfig,
    BertForNextSentencePrediction,
    BertModel,
    BertTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)

from get_dataset import get_dd_corpus, get_zhao_dataset
from trainer import Trainer
from utils import (
    dump_config,
    eval_by_NSP,
    get_correlation,
    get_logger,
    load_model,
    save_model,
    set_random_seed,
    write_summary,
)


def main():
    def save_prediction(fname, human, model_res, model_name):
        assert len(human) == len(model_res)
        saved = []
        for idx in range(len(human)):
            saved.append({"score": human[idx], model_name: model_res[idx]})
        with open(fname, "w") as f:
            for l in saved:
                json.dump(l, f)
                f.write("\n")

    dataset_name = "dd"
    assert dataset_name in ["dd", "persona"]
    result_save_path = "baselines"
    os.makedirs(result_save_path, exist_ok=True)
    fname_format = result_save_path + "/{}_{}.jsonl"  # metric_name, setname

    zhao_dataset = get_zhao_dataset(dataset_name)
    print(zhao_dataset[0])

    """
    BERTSCORE
    """
    print("BERTSCORE")
    bertscore_humanscore, bertscore_prediction = [], []
    hyp_list, ref_list = [], []
    for item in tqdm(zhao_dataset):
        ref, hyp, humanscore = (
            item["ref"],
            item["hyp"],
            item["human_score"],
        )
        hyp_list.append(hyp)
        ref_list.append(ref)
        bertscore_humanscore.append(humanscore)
    bertscore_prediction = [
        float(el) for el in score(hyp_list, ref_list, rescale_with_baseline=True, lang="en")[-1]
    ]
    assert len(bertscore_prediction) == len(bertscore_humanscore)
    p, s = get_correlation(bertscore_humanscore, bertscore_prediction)
    print("Pearson: {}".format(p))
    print("Spearman: {}\n".format(s))
    save_prediction(
        fname_format.format("bertscore", dataset_name),
        bertscore_humanscore,
        bertscore_prediction,
        "bertscore",
    )

    """
    NLGEval
    """
    nlgeval = NLGEval(metrics_to_omit=["SkipThoughtCS"])
    recorder = []
    for item in tqdm(zhao_dataset):
        ctx, ref, hyp, humanscore = (
            item["ctx"],
            item["ref"],
            item["hyp"],
            item["human_score"],
        )

        res = nlgeval.compute_individual_metrics([ref], hyp)
        res["score"] = humanscore
        recorder.append(res)
    with open(fname_format.format("nlgeval", dataset_name), "w") as f:
        for l in recorder:
            json.dump(l, f)
            f.write("\n")


if __name__ == "__main__":
    main()