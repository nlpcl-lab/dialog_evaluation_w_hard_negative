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

from pprint import pprint
from get_dataset import get_dd_corpus, get_zhao_dataset, get_persona_corpus
from utils import (
    dump_config,
    get_correlation,
    draw_scatter_plot,
    load_model,
    save_model,
    softmax_2d,
    set_random_seed,
    get_usencoder,
    write_summary,
)
from nlgeval import NLGEval
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from bert_score import score as bertscore
from nltk.translate.bleu_score import sentence_bleu
from matplotlib import pyplot as plt
import pickle
from retrieve_with_bi_encoder import make_annotated_dataset

import argparse


def main(args):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    special_tokens_dict = {"additional_special_tokens": ["[SEPT]"]}
    tokenizer.add_special_tokens(special_tokens_dict)

    if args.eval_set == 'dd':
        raw_dd_train = get_dd_corpus("train")
        test_data = make_annotated_dataset(get_zhao_dataset("dd"), tokenizer)
    else:
        raw_dd_train = get_persona_corpus("train")
        test_data = make_annotated_dataset(get_zhao_dataset("persona"), tokenizer)

    all_uttrs = [uttr for conv in raw_dd_train for uttr in conv]

    with open(args.ir_result_json_fname, "rb") as f:
        response_score = np.load(f)
    #    with open("./data/cache/dd-hyp-bi64-epoch0-1000.json", "rb") as f:
    #        hyp_score = np.load(f)

    use_model = get_usencoder()

    nlgeval = NLGEval(
        metrics_to_omit=[
            "SkipThoughtCS",
            "CIDEr",
            "Bleu_3",
            "Bleu_4",
        ]
    )

    def get_similarity_score(metric: str, ref, hyp: str):
        assert isinstance(ref, list) and isinstance(hyp, str)
        if metric == "use":
            ref_emb_list = use_model(ref)
            hyp_emb_list = use_model([hyp])
            sim_list = cosine_similarity(hyp_emb_list, ref_emb_list)[0]
            assert len(sim_list) == len(ref)
            return {"use": [float(el) for el in sim_list]}
        elif metric == "bertscore":
            return [
                float(el)
                for el in bertscore(
                    [hyp for _ in range(len(ref))], ref, rescale_with_baseline=True, lang="en"
                )[-1]
            ]
        elif metric == "nlgeval":
            result = [nlgeval.compute_individual_metrics([el], hyp) for el in ref]
            sim_avg = [el["EmbeddingAverageCosineSimilarity"] for el in result]
            sim_grd = [el["GreedyMatchingScore"] for el in result]
            sim_ext = [el["VectorExtremaCosineSimilarity"] for el in result]
            sim_meteor = [el["METEOR"] for el in result]
            sim_rouge = [el["ROUGE_L"] for el in result]
            sim_b1 = [el["Bleu_1"] for el in result]
            sim_b2 = [el["Bleu_2"] for el in result]
            return {
                "avg": sim_avg,
                "grd": sim_grd,
                "ext": sim_ext,
                "meteor": sim_meteor,
                "rouge": sim_rouge,
                "bleu1": sim_b1,
                "bleu2": sim_b2,
            }

    similarity_list = {}

    for metric in [
        "nlgeval",
        "use",
    ]:
        print(metric)
        retrieve_num = args.max_candidate

        sorted_response_index = np.argsort(-response_score, 1)[:, :retrieve_num]
        sorted_response_uttrs = [
            [all_uttrs[idx] for idx in one_indices] for one_indices in sorted_response_index
        ]

        humanscore = []

        for idx, item in enumerate(tqdm(test_data)):
            ctx, ref, hyp, score = item
            # prepare input
            humanscore.append(score)
            hyp = " ".join(tokenizer.convert_ids_to_tokens(hyp[1:-1])).replace(" ##", "")
            ref = " ".join(tokenizer.convert_ids_to_tokens(ref[1:-1])).replace(" ##", "")
            retrieved_candidate = sorted_response_uttrs[idx]

            # get similarity
            hyp_ref_score = get_similarity_score(metric, [ref], hyp)
            candidate_score = get_similarity_score(metric, retrieved_candidate, hyp)

            # save the similarity
            for k, v in hyp_ref_score.items():
                if k not in similarity_list:
                    similarity_list[k] = {}
                if "ref" not in similarity_list[k]:
                    similarity_list[k]["ref"] = []
                assert len(v) == 1
                similarity_list[k]["ref"].append(v[0])
            for k, v in candidate_score.items():
                if k not in similarity_list:
                    similarity_list[k] = {}
                if "similarity" not in similarity_list[k]:
                    similarity_list[k]["similarity"] = []
                assert len(v) == retrieve_num
                similarity_list[k]["similarity"].append(v)

        similarity_list["human"] = humanscore

    os.makedirs(os.path.dirname(args.output_fname), exist_ok=True)
    with open(args.output_fname, "wb") as f:
        pickle.dump(similarity_list, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--ir_method", type=str, default="bi_encoder", choices=["bi_encoder", "bm25"])
    parser.add_argument("--eval_set", type=str, default="dd", choices=["dd", "persona"])
    parser.add_argument(
        "--ir_result_json_fname",
        type=str,
        default="./data/cache/dd-response-bi64-epoch0-1000.json",
        choices=["./data/cache/dd-response-bi64-epoch0-1000.json",'./data/cache/persona-response-bi64-epoch0-1000.json'],
    )
    parser.add_argument("--max_candidate", type=int, default=500)
    args = parser.parse_args()

    args.output_fname = "./ours_data/{}-{}-{}-similarity.pck".format(
        args.eval_set, args.ir_method, args.max_candidate
    )

    main(args)