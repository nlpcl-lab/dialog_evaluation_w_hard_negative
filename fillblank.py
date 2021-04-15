"""
Golden response를 constituency parsing 한 다음, pre-order traversal 하면서 subtree를 drop.
Dropped leaves는 마스킹하고, context 없는 상태에서 BART로 다시 infilling.
Reconstructed response는 USE 등을 통해 further ranking.
"""

import json
import os
from typing import List, Tuple

import allennlp_models.tagging
import nltk
import torch
from allennlp.predictors.predictor import Predictor
from tqdm import tqdm
from transformers import BartForConditionalGeneration, BartTokenizer


def get_bart() -> Tuple[BartForConditionalGeneration, BartTokenizer]:
    pass


def get_allen_consituency_parser() -> Predictor:
    return Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz"
    )


def read_random_dataset(fname: str) -> List[List[str]]:
    assert os.path.exists(fname)
    with open(fname, "r") as f:
        ls = [el.strip().split("|||") for el in f.readlines()]
        assert all([len(el) == 3] for el in ls)
        return ls


def parse_constituency(config, setname: str):
    """Pasing the golden response and save the result with some additional post-processing"""
    raw_dataset = read_random_dataset(setname)

    parser = get_allen_consituency_parser()


def masking(config, setname: str):
    """Maksing the parsed tree with the given constraints"""
    pass


def infill_by_bart(config, setname: str):
    """Infill the masked golden response with generative LM (BART)"""
    pass


def rank_reconstructed_rseponse(config, setname: str):
    """Rank the reconstructed golden responses by BART using ???."""
    pass


def main(config):
    for setname in ["valid", "train"]:
        parse_constituency(config, setname)
        masking(config, setname)
        infill_by_bart(config, setname)
        rank_reconstructed_rseponse(config, setname)


class Config:
    random_fname = "./data/negative/random_neg1_{}.txt"
    parse_save_fname = "./parsed/consti_{}.json"


if __name__ == "__main__":
    config = Config()
    main(config)
