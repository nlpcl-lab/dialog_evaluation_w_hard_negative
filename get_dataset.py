import json
import os
import wget
import zipfile
from typing import List, Union
from google_drive_downloader import GoogleDriveDownloader as gdd


DAILYDIALOG_URL = "http://yanran.li/files/ijcnlp_dailydialog.zip"
annotated_dailydialog_drive_id: str = "1tbSnH20B2SRBeqiZTiW7NKhE_EVL6ktw"
annotated_persona_drive_id: str = "1eTC-xMxz4P-s7B1UGBcqwnJZTjVdaaF2"


def download_dailydialog(daily_raw_fname: str):
    os.makedirs("data", exist_ok=True)
    """Download the raw DailyDialog dataset
    Args:
        daily_raw_fname (str): Raw DailyDialog dataset URL
        data_path (str): Path to save
    """
    wget.download(daily_raw_fname, "data")
    # Manually unzip the train/dev/test files


def get_grade_annotated_dataset(corpus_name: str):
    assert corpus_name in ["dailydialog", "convai2", "empatheticdialogues"]
    dialog_path = "./data/GRADE/evaluation/eval_data/{}/".format(corpus_name)
    score_path = "./data/GRADE/evaluation/human_score/{}/".format(corpus_name)
    model_list = os.listdir(dialog_path)
    result = []

    for model_name in model_list:
        text_path = os.path.join(dialog_path, model_name)
        score_fname = score_path + "{}/human_score.txt".format(model_name)
        ctx_fname = os.path.join(text_path, "human_ctx.txt")
        ref_fname = os.path.join(text_path, "human_ref.txt")
        hyp_fname = os.path.join(text_path, "human_hyp.txt")

        scores, ctxs, refs, hyps = [
            _read_txt_files(fname)
            for fname in [score_fname, ctx_fname, ref_fname, hyp_fname]
        ]
        # assert the same number of samples
        assert len(list(set(list(map(len, [scores, ctxs, refs, hyps]))))) == 1
        # assert uniquness of score,ref and hyp
        assert all([len(el) == 1 for el in scores])
        assert all([len(el) == 1 for el in refs])
        assert all([len(el) == 1 for el in hyps])
        scores, refs, hyps = (
            [(float(el[0]) - 1) / 4 for el in scores],
            [el[0] for el in refs],
            [el[0] for el in hyps],
        )

        for idx in range(len(scores)):
            assert 0 <= scores[idx] <= 1
            result.append(
                {
                    "corpus": corpus_name,
                    "model": model_name,
                    "human_score": scores[idx],
                    "ctx": ctxs[idx],
                    "ref": refs[idx],
                    "hyp": hyps[idx],
                }
            )
    return result


def _read_txt_files(fname: str):
    assert os.path.exists(fname)
    with open(fname, "r") as f:
        ls = [el.strip().split("|||") for el in f.readlines()]
    return ls


def download_Zhao_dataset(
    daily_id: str,
    daily_output_fname: str,
    persona_id: str,
    persona_output_fname: str,
):
    """Download the dataset with human-annotated score by Zhao et al., ACL 2020
    Args:
        daily_id (str): Annotated DailyDialog dataset ID
        daily_output_fname (str): Path for output
        persona_id (str): Annotated Personachat dataset ID
        persona_output_fname (str): Path for output
    """
    gdd.download_file_from_google_drive(
        daily_id, daily_output_fname, unzip=False
    )
    gdd.download_file_from_google_drive(
        persona_id, persona_output_fname, unzip=False
    )


def get_zhao_dataset(fname):
    assert fname in ["dd", "persona"]
    with open("./data/zhao/{}.json".format(fname), "r") as f:
        ls = list(json.load(f).values())

    dataset = []

    for item in ls:
        assert all([len(el) == 2 for el in item["context"]])
        assert len(item["reference"]) == 2
        context = [el[1].lower().strip() for el in item["context"]]

        reference = item["reference"][1].lower()
        responses = item["responses"]
        for k, v in responses.items():
            # skip the ground-truth for quality estimation
            if k == "ground-truth":
                assert v["uttr"] == reference
                continue

            scores = [el["overall"] for _, el in v["scores"].items()]
            scores = sum(scores) / len(scores)

            assert 1 <= scores <= 5
            dataset.append(
                {
                    "ctx": context,
                    "ref": reference,
                    "hyp": v["uttr"].lower(),
                    "model": k,
                    "human_score": scores,
                    "corpus": fname,
                }
            )
    from pprint import pprint

    pprint(dataset[-1])

    return dataset


def get_dd_corpus(setname):
    assert setname in ["train", "validation", "test"]
    fname = "./data/ijcnlp_dailydialog/{}/dialogues_{}.txt".format(
        setname, setname
    )
    assert os.path.exists(fname)
    with open(fname, "r") as f:
        ls = [el.strip() for el in f.readlines()]
        for idx, line in enumerate(ls):
            line = [
                el.strip().lower()
                for el in line.split("__eou__")
                if el.strip() != ""
            ]
            ls[idx] = line
    return ls


def main():
    """
    download_dailydialog(DAILYDIALOG_URL, "./data/")
    os.makedirs("./data/zhao/", exist_ok=True)
    download_Zhao_dataset(
        annotated_dailydialog_drive_id,
        "./data/zhao/dd.json",
        annotated_persona_drive_id,
        "./data/zhao/persona.json",
    )
    """

    # Manually download the GRADE dataset at './data/'
    get_grade_annotated_dataset("dailydialog")
    get_grade_annotated_dataset("convai2")
    get_grade_annotated_dataset("empatheticdialogues")

    res = get_zhao_dataset("dd")
    res = get_zhao_dataset("persona")
    res = get_dd_corpus("train")
    res = get_dd_corpus("validation")
    res = get_dd_corpus("test")
