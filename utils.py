import os, json
import torch
import random
import numpy as np
from scipy.stats import pearsonr, spearmanr
from typing import List
import logging


def set_random_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def dump_config(args):
    with open(os.path.join(args.exp_path, "config.json"), "w") as f:
        json.dump(vars(args), f)


def save_model(model, path, epoch):
    torch.save(
        model.state_dict(),
        os.path.join(path, f"epoch-{epoch}.pth"),
    )


def load_model(model, path, epoch, len_tokenizer):
    model.resize_token_embeddings(len_tokenizer)
    model.load_state_dict(torch.load(path + f"/epoch-{epoch}.pth"))
    return model


def write_summary(writer, values, setname: str, step: int):
    for k, v in values.items():
        writer.add_scalars(k, {setname: v}, step)
    writer.flush()


def get_correlation(human_score: List[float], model_score: List[float]):
    p = pearsonr(human_score, model_score)
    s = spearmanr(human_score, model_score)
    return p[0], s[0]


def eval_by_NSP(dataset, model, device):
    recorder = []
    softmax = torch.nn.Softmax(dim=1)
    feature = dataset.feature

    for batch in feature:
        ids, types, masks, scores = (
            batch["input_ids"].to(device),
            batch["token_type_ids"].to(device),
            batch["attention_mask"].to(device),
            torch.tensor(batch["human_score"]).to(device),
        )
        with torch.no_grad():
            prediction = (
                softmax(model(ids, masks, types)[0]).cpu().numpy()[0][0]
            )
        recorder.append(
            {"human_score": batch["human_score"], "nsp": prediction}
        )
    return recorder


def get_logger() -> logging.Logger:
    """Return the Logger class"""
    # create logger
    logger = logging.getLogger("MyLogger")
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)
    logger.info("Logger Generated")
    return logger


def read_raw_file(fname: str):
    with open(fname, "r") as f:
        ls = [el.strip().split("|||") for el in f.readlines()]
    return ls
