import networkx as nx
from gensim.models import KeyedVectors
import random
from nltk.tokenize import word_tokenize
from get_dataset import get_dd_corpus, get_zhao_dataset
from utils import read_raw_file
from gensim.scripts.glove2word2vec import glove2word2vec
import os
import numpy as np


KEYWORD_FILE = "./data/keyword.vocab"


def get_keyword_list():
    with open(KEYWORD_FILE, "r") as f:
        return [el.strip() for el in f.readlines()]


keyword_list = get_keyword_list()


def get_keyword_from_utterance(uttr, keywords=keyword_list):
    words = word_tokenize(uttr.lower())
    return [el for el in words if el in keywords]


def change_glove_2_w2v():
    glove_input_file = "./tools/numberbatch-en-19.08.txt"
    word2vec_output_file = "./tools/numberbatch-en-19.08.word2vec.txt"
    (count, dimensions) = glove2word2vec(glove_input_file, word2vec_output_file)


def load_cpnet():
    print("Load cpnet...")
    cpnet = nx.read_gpickle("./tools/cpnet.graph")
    print("Done")

    cpnet_simple = nx.MultiDiGraph()
    for u, v, data in cpnet.edges(data=True):
        w = data["weight"] if "weight" in data else 1.0
        if cpnet_simple.has_edge(u, v):
            continue
        else:
            cpnet_simple.add_edge(u, v)

    return cpnet_simple


def load_resources():
    word2vec_model_path = "./tools/numberbatch-en-19.08.word2vec.txt"
    if not os.path.exists(word2vec_model_path):
        change_glove_2_w2v()
        assert os.path.exists(word2vec_model_path)

    word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model_path)

    concept2id = {}
    id2concept = {}
    with open("./tools/concept.txt", "r", encoding="utf8") as f:
        for w in f.readlines():
            if len(concept2id) == 50000:
                break
            concept2id[w.strip()] = len(concept2id)
            id2concept[len(id2concept)] = w.strip()

    print("concept2id done")
    id2relation = {}
    relation2id = {}
    with open("./tools/relation.txt", "r", encoding="utf8") as f:
        for w in f.readlines():
            id2relation[len(id2relation)] = w.strip()
            relation2id[w.strip()] = len(relation2id)
    print("relation2id done")

    embeddings = []
    for c in concept2id:
        if c not in word2vec_model:
            vec = np.random.rand(1, 300)[0]
        else:
            vec = word2vec_model.get_vector(c)
        embeddings.append(vec)
    assert len(embeddings) == 50000
    embeddings = np.array(embeddings)

    return concept2id, id2concept, relation2id, id2relation, embeddings


def get_one_hop_neighborhoods(cpnet, concept2id, id2concept, keywords_in_dialog, k):
    assert all([isinstance(e, str) for e in keywords_in_dialog])
    neighbors = []
    for keyword in keywords_in_dialog:
        neigh = list(cpnet.neighbors(concept2id[keyword]))
        neigh = random.sample(neigh, min(neigh, k))
        neigh = [id2concept[el] for el in neigh]
        neighbors.append(neigh)
    return neighbors


def get_distance_btw_two_keyword(cpnet, concept2id, src, tgt):
    if src not in concept2id or tgt not in concept2id:
        return -1
    src_id, tgt_id = concept2id[src], concept2id[tgt]
    if src_id not in cpnet.nodes() or tgt_id not in cpnet.nodes():
        return -1
    try:
        return (
            nx.shortest_path_length(cpnet, source=src_id, target=tgt_id) + 1
        )  # compute shortest path
    except:
        return -1