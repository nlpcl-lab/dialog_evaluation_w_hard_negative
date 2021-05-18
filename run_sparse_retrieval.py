import os
from get_dataset import get_dd_corpus, get_persona_corpus
import argparse
import json
from rank_bm25 import BM25Okapi


def make_feature(eval_set, output_dir):
    raw_data = get_dd_corpus("train") if eval_set == "dd" else get_persona_corpus("validation")
    featured = make_docs(raw_data)
    serialize_feature(featured, output_dir)


def serialize_feature(featured, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fname = os.path.join(output_dir, f"docs.jsonl")
    with open(fname, "w") as f:
        for l in featured:
            f.write(json.dumps(l) + "\n")


def make_docs(raw_data):
    datas = []
    for conv in raw_data:
        for idx in range(len(conv[:-1])):
            uttrs = conv[: idx + 2]
            context, reply = " ".join(uttrs[:-1]), uttrs[-1]
            datas.append({"id": "doc" + str(1 + len(datas)), "contents": context, "reply": reply})
    return datas


def search_example(setname):
    from pyserini.search import SimpleSearcher

    searcher = SimpleSearcher("indexes/{}_index.jsonl".format(setname))
    hits = searcher.search("How are you?")

    for i in range(len(hits)):
        print(f"{i+1:2} {hits[i].docid:15} {hits[i].score:.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--eval_set", type=str, default="dd", choices=["dd", "persona"])

    args = parser.parse_args()
    args.document_dir = f"{args.eval_set}-docs"
    make_feature(args.eval_set, args.document_dir)

    # search_example(args.eval_set)
