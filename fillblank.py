"""
Golden response를 constituency parsing 한 다음, pre-order traversal 하면서 subtree를 drop.
Dropped leaves는 마스킹하고, context 없는 상태에서 BART로 다시 infilling.
Reconstructed response는 USE 등을 통해 further ranking.
"""

import json
import os
from typing import Dict, List, Tuple

import allennlp_models.tagging
import nltk
import numpy as np
import torch
from allennlp.predictors.predictor import Predictor
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import BartForConditionalGeneration, BartTokenizer

from utils import set_random_seed

MASK_TOKEN = "<mask>"


def get_bart(device) -> Tuple[BartForConditionalGeneration, BartTokenizer]:
    model = BartForConditionalGeneration.from_pretrained(
        "facebook/bart-large", force_bos_token_to_be_generated=True
    )
    tok = BartTokenizer.from_pretrained("facebook/bart-large")
    model.eval()
    model.to(device)
    return model, tok


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


def read_jsonl(fname):
    with open(fname, "r") as f:
        return [json.loads(el) for el in f.readlines()]


def add_indices_to_terminals(tree: nltk.tree.Tree) -> nltk.tree.Tree:
    """append the absolute token index to each leaves(token) in the tree for further process"""
    for idx, _ in enumerate(tree.leaves()):
        tree_location = tree.leaf_treeposition(idx)
        non_terminal = tree[tree_location[:-1]]
        non_terminal[0] = non_terminal[0] + f"__[{idx}]"
    return tree


def parse_constituency(config, setname: str) -> List[Dict]:
    """Pasing the golden response and save the result with some additional post-processing"""
    print(f"--- Constituency Parsing for {setname} ---")
    save_fname = config.parse_save_fname.format(setname)
    if os.path.exists(save_fname):
        print("Already exist!")
        return
    raw_dataset = read_random_dataset(config.random_fname.format(setname))
    parser = get_allen_consituency_parser()

    saver = []
    os.makedirs(os.path.dirname(save_fname), exist_ok=True)
    for item_idx, item in enumerate(tqdm(raw_dataset)):
        context, golden, rand_neg = item
        parsed_golden = parser.predict(golden)["trees"]
        # tree = nltk.tree.Tree.fromstring(parsed_golden)
        # tree_with_index = add_indices_to_terminals(tree)
        item = {"idx": item_idx, "raw_triple": item, "treestring": parsed_golden}
        saver.append(item)
    with open(save_fname, "w") as f:
        for line in saver:
            json.dump(line, f)
            f.write("\n")
    return saver


def masking(config, setname: str):
    """Maksing the parsed tree with the given constraints"""
    print(f"--- Tree masking for {setname} ---")
    output_fname = config.masked_save_fname.format(setname)
    if os.path.exists(output_fname):
        print("Already exist!")
        return

    parsed_data = read_jsonl(config.parse_save_fname.format(setname))
    saver = []
    masked_ratio = []
    success_counter = []
    for item_idx, item in enumerate(tqdm(parsed_data)):
        assert item_idx == item["idx"]
        masked_leaves = []
        triple, parsed_tree = item["raw_triple"], add_indices_to_terminals(
            nltk.tree.Tree.fromstring(item["treestring"])
        )
        total_leaves = parsed_tree.leaves()
        if len(total_leaves) < config.original_minimum_len:
            item["masked"] = ""
            saver.append(item)
            success_counter.append(False)
            continue

        def traverse_and_mask(tree):
            """
            Tree를 pre-order traversal하면서 각 node 및 subtree를 p % 확률로 지움.
            한번에 너무 많이 지우는걸 박음.
            """
            # Do mask this node or not?
            leaves = tree.leaves()
            is_mask = (
                bool(np.random.binomial(1, config.tree_mask_prob))
                and (len(leaves) / len(total_leaves) < config.subtree_mask_max_ratio)
                and not (leaves == total_leaves[: len(leaves)])
                and not (
                    len(leaves) == 1
                    and leaves[0][0] in ".,!?"
                    and leaves[0][1:3] == "__"
                )
            )
            if is_mask:
                masked_leaves.extend(tree.leaves())
                return
            for subtree in tree:
                if type(subtree) == nltk.tree.Tree:
                    traverse_and_mask(subtree)

        success = False
        patient = 0
        while True:
            if patient == config.num_patient_for_mask:
                break
            traverse_and_mask(parsed_tree)
            assert len(set(masked_leaves)) == len(masked_leaves)
            assert set(masked_leaves).issubset(set(total_leaves))
            # Constraint list
            minnum_const = (
                len(masked_leaves) / len(total_leaves) > config.min_mask_ratio
            )
            maxnum_const = (
                len(masked_leaves) / len(total_leaves) < config.max_mask_ratio
            )
            if all([minnum_const, maxnum_const]):
                success = True
                break
            # Retry until meet the list of constraints...
            masked_leaves = []
            patient += 1
        masked_ratio.append(len(masked_leaves) / len(total_leaves))
        item["masked"] = masked_leaves if success else ""
        if success:
            corrupted_golden = corrupted_token_to_string(total_leaves, masked_leaves)

        success_counter.append(success)
        saver.append(item)
    print("Masking ratio: ", sum(masked_ratio) / len(masked_ratio))
    print("Masking Success ratio: ", sum(success_counter) / len(success_counter))
    assert len(saver) == len(parsed_data) == len(success_counter)

    with open(output_fname, "w") as f:
        for l in saver:
            json.dump(l, f)
            f.write("\n")


def corrupted_token_to_string(
    tokenized_golden: List[str], token_tobe_masked: List[str]
) -> str:
    corrupted_golden = [
        MASK_TOKEN if el in token_tobe_masked else el for el in tokenized_golden
    ]
    input_seq = ""
    # remove the token position identifier ('__[IDX]') and merge the consecutive <mask> into once.
    for tok in corrupted_golden:
        assert "__[" in tok or MASK_TOKEN == tok
        if tok != MASK_TOKEN:
            org_tok = tok.split("__[")
            assert len(org_tok) == 2
            org_tok = org_tok[0]
            input_seq += " " + org_tok
            continue
        assert tok == MASK_TOKEN
        if (
            len(input_seq) < len(MASK_TOKEN)
            or input_seq[-len(MASK_TOKEN) :] != MASK_TOKEN
        ):
            input_seq += " " + MASK_TOKEN
        assert input_seq[-len(MASK_TOKEN) :] == MASK_TOKEN

    return input_seq.strip()


def infill_by_bart(config, setname: str):
    """Infill the masked golden response with generative LM (BART)"""
    print(f"--- Infilling by BART for {setname} ---")

    device = torch.device("cuda")
    output_fname = config.bart_gen_fname.format(setname)
    if os.path.exists(output_fname):
        exist_data = read_jsonl(output_fname)
        assert exist_data[0]["mask_fname"] == config.masked_save_fname.format(setname)
        print("Already exist!")
        return
    masked_data = read_jsonl(config.masked_save_fname.format(setname))
    bart, tokenizer = get_bart(device)
    detokenizer = TreebankWordDetokenizer()
    saver = []
    for item_idx, item in enumerate(tqdm(masked_data)):
        item["mask_fname"] = config.masked_save_fname.format(setname)
        if "masked" not in item or len(item["masked"]) == 0:
            item["generated"] = []
            item["input_seq"] = ""
            saver.append(item)
            continue

        token_tobe_masked = item["masked"]
        original_response = item["raw_triple"][1]
        tokenized_golden = add_indices_to_terminals(
            nltk.tree.Tree.fromstring(item["treestring"])
        ).leaves()

        input_seq = corrupted_token_to_string(tokenized_golden, token_tobe_masked)
        input_seq = input_seq[0].upper() + input_seq[1:]
        input_seq = detokenizer.detokenize(input_seq.strip().split())
        input_seq = [el.strip() for el in input_seq.split(MASK_TOKEN)]
        input_seq = f" {MASK_TOKEN} ".join(input_seq)

        # Generate!
        batch = tokenizer(input_seq, return_tensors="pt")
        with torch.no_grad():
            generated_ids = bart.generate(
                batch["input_ids"].to(device),
                max_length=128,
                # num_beams=5,
                bad_words_ids=[[4], [6], [328], [116]],
                do_sample=config.do_sample,
                top_p=config.top_p,
                no_repeat_ngram_size=3,
                num_return_sequences=config.num_ret_seq,
            )
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        assert all([isinstance(el, str) for el in decoded])
        item["input_seq"] = input_seq
        print(generated_ids)
        item["generated"] = decoded
        saver.append(item)

        print("-" * 50)
        print("Golden: ", original_response)
        print("Input seq: ", input_seq)
        print("Generated: ", decoded[0])
    # assert len(saver) == len(masked_data)

    with open(output_fname, "w") as f:
        for l in saver:
            json.dump(l, f)
            f.write("\n")


def fileter_reconstructed_negative(config, setname: str):
    """Rank the reconstructed golden responses using ???."""
    output_fname = config.negative_output_fname.format(setname)
    # assert not os.path.exists(output_fname)
    from utils import get_usencoder

    use_model = get_usencoder()
    with open(config.bart_gen_fname.format(setname), "r") as f:
        ls = [json.loads(el) for el in f.readlines()]
    filtered = []
    final_output = []
    for idx, item in enumerate(tqdm(ls)):
        triple = item["raw_triple"]
        generated = item["generated"]
        if len(generated) == 0:
            triple.append("[NONE]")
            final_output.append(triple)
            continue
        generated = generated[0]
        """
        LENGTH
        """
        if len(generated) > 2 * len(triple[1]):
            triple.append("[NONE]")
            final_output.append(triple)
            continue

        """
        USE
        """
        generated = " ".join(word_tokenize(generated.lower()))
        golden_emb = use_model([triple[1]])
        generated_emb = use_model([generated])
        assert len(golden_emb) == len(generated_emb) == 1
        cossim = cosine_similarity(golden_emb, generated_emb)
        assert len(cossim) == 1 and len(cossim[0]) == 1
        cossim = float(cossim[0][0])

        if cossim > config.use_encoder_threshold:
            filtered.append(True)
            generated = "[NONE]"
        else:
            filtered.append(False)

        """
        Postprocessing
        """
        for adhoc_tok in ["''", "'", ":", "...", ".."]:
            if adhoc_tok in generated and adhoc_tok not in triple[1]:
                generated = generated.replace(adhoc_tok, "")
        triple.append(generated)
        final_output.append(triple)
    print("Filtered: ", sum(filtered) / len(filtered))
    with open(output_fname, "w") as f:
        for line in final_output:
            f.write("|||".join(line))
            f.write("\n")


def main(config):
    for setname in ["valid", "train"]:
        parsed = parse_constituency(config, setname)
        masking(config, setname)
        infill_by_bart(config, setname)
        fileter_reconstructed_negative(config, setname)


class Config:
    random_fname = "./data/negative/random_neg1_{}.txt"
    parse_save_fname = "./parsed/consti_{}.json"

    tree_mask_prob: float = 0.2
    subtree_mask_max_ratio: float = 0.4
    original_minimum_len: int = 10
    num_patient_for_mask: int = 10
    min_mask_ratio: float = 0.3
    max_mask_ratio: float = 0.5
    masked_save_fname = (
        f"./parsed/mask_prob{tree_mask_prob}_subtreelimit{subtree_mask_max_ratio}_maskmin{min_mask_ratio}_maskmax{max_mask_ratio}_orgminlen{original_minimum_len}_patient{num_patient_for_mask}"
        + "_{}.json"
    )

    do_sample: bool = True
    top_p = 0.95
    top_k = 100
    num_ret_seq: int = 1
    # bart_gen_fname = f"./parsed/bart_topk{top_k}_ret{num_ret_seq}" + "_{}.json"
    # bart_gen_fname = f"./parsed/bart_beam5" + "_{}.json"
    bart_gen_fname = f"./parsed/bart_topp{top_p}" + "_{}.json"
    if do_sample:
        assert "topp" in bart_gen_fname
    else:
        assert "beam" in bart_gen_fname

    negative_output_fname = "./data/negative/" + bart_gen_fname.split("/")[-1].replace(
        ".json", ".txt"
    )
    use_encoder_threshold = 0.9


if __name__ == "__main__":
    config = Config()
    set_random_seed()
    main(config)
