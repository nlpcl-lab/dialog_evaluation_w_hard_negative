import json
import logging
import os
import random
from typing import List

from matplotlib import pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm


def compute_mask_indices(
    shape,
    padding_mask,
    mask_prob: float,
    mask_length: int,
    mask_type: str = "static",
    mask_other: float = 0.0,
    min_masks: int = 0,
    no_overlap: bool = False,
    min_space: int = 0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape
    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        padding_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_type: how to compute mask lengths
            static = fixed size
            uniform = sample from uniform distribution [mask_other, mask_length*2]
            normal = sample from normal distribution with mean mask_length and stdev mask_other. mask is min 1 element
            poisson = sample from possion distribution with lambda = mask length
        min_masks: minimum number of masked spans
        no_overlap: if false, will switch to an alternative recursive algorithm that prevents spans from overlapping
        min_space: only used if no_overlap is True, this is how many elements to keep unmasked between spans
    """

    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    all_num_mask = int(
        # add a random number for probabilistic rounding
        mask_prob * all_sz / float(mask_length)
        + np.random.rand()
    )

    all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []
    for i in range(bsz):
        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            num_mask = int(
                # add a random number for probabilistic rounding
                mask_prob * sz / float(mask_length)
                + np.random.rand()
            )
            num_mask = max(min_masks, num_mask)
        else:
            sz = all_sz
            num_mask = all_num_mask

        if mask_type == "static":
            lengths = np.full(num_mask, mask_length)
        elif mask_type == "uniform":
            lengths = np.random.randint(mask_other, mask_length * 2 + 1, size=num_mask)
        elif mask_type == "normal":
            lengths = np.random.normal(mask_length, mask_other, size=num_mask)
            lengths = [max(1, int(round(x))) for x in lengths]
        elif mask_type == "poisson":
            lengths = np.random.poisson(mask_length, size=num_mask)
            lengths = [int(round(x)) for x in lengths]
        else:
            raise Exception("unknown mask selection " + mask_type)

        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)

        if no_overlap:
            mask_idc = []

            def arrange(s, e, length, keep_length):
                span_start = np.random.randint(s, e - length)
                mask_idc.extend(span_start + i for i in range(length))

                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - keep_length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts

            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                lens = np.fromiter(
                    (e - s if e - s >= length + min_space else 0 for s, e in parts),
                    np.int,
                )
                l_sum = np.sum(lens)
                if l_sum == 0:
                    break
                probs = lens / np.sum(lens)
                c = np.random.choice(len(parts), p=probs)
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = np.asarray(mask_idc)
        else:
            min_len = min(lengths)
            if sz - min_len <= num_mask:
                min_len = sz - num_mask - 1

            mask_idc = np.random.choice(sz - min_len, num_mask, replace=False)

            mask_idc = np.asarray(
                [mask_idc[j] + offset for j in range(len(mask_idc)) for offset in range(lengths[j])]
            )

        mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))

    min_len = min([len(m) for m in mask_idcs])
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        mask[i, mask_idc] = True

    return mask


def get_usencoder():
    import tensorflow_hub as hub

    return hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


def set_random_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def draw_scatter_plot(title, x, y, fname):
    x = [el + np.random.normal(0, 0.3 * 0.3, 1)[0] for el in x]
    fig, axs = plt.subplots()
    axs.set_title(title)
    axs.set_title(title)
    axs.set_xticks([1, 2, 3, 4, 5])
    axs.scatter(x, y, s=0.3, c="green")

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=300)
    plt.clf()


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


def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x


def softmax_2d(x):
    f_x = np.exp(x) / np.sum(np.exp(x), 1)[:, np.newaxis]
    return f_x


def eval_by_NSP(dataset, model, device, is_rank: bool = False):
    recorder = []
    softmax = torch.nn.Softmax(dim=1)
    feature = dataset.feature

    for batch in tqdm(feature):
        if is_rank:
            context, response = batch["ctx"], batch["hyp"]
            ctx_ids, ctx_mask = (
                context["input_ids"].to(device),
                context["attention_mask"].to(device),
            )
            hyp_ids, hyp_mask = (
                response["input_ids"].to(device),
                response["attention_mask"].to(device),
            )
            with torch.no_grad():
                prediction = model(ctx_ids, ctx_mask, hyp_ids, hyp_mask).cpu().numpy()[0][0]

        else:
            ids, types, masks, scores = (
                batch["input_ids"].to(device),
                batch["token_type_ids"].to(device),
                batch["attention_mask"].to(device),
                torch.tensor(batch["human_score"]).to(device),
            )
            with torch.no_grad():
                # prediction = (
                #    softmax(model(ids, masks, types)[0]).cpu().numpy()[0][0]
                # )
                prediction = softmax(model(ids, masks)[0]).cpu().numpy()[0][0]
        recorder.append({"human_score": batch["human_score"], "nsp": prediction})
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
