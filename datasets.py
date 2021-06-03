import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm import tqdm
import numpy as np


from utils import read_raw_file
from graph_preprocess import *

TURN_TOKEN = "[SEPT]"


class GraphNSPDataset(Dataset):
    def __init__(self, fname, max_seq_len: int, tokenizer, cpnet, c2i, i2c, w2v, max_keyword_num):
        self.max_keyword_num = max_keyword_num
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.fname = fname
        self.cpnet = cpnet
        self.c2i = c2i
        self.i2c = i2c
        self.raw_data = self.read_dataset(fname)
        self.feature = self._make_feature(self.raw_data)

    def read_dataset(self, fname):
        raw = read_raw_file(fname)
        assert all([len(el) == 3 for el in raw])
        return raw

    def __len__(self):
        assert len(self.feature[0]) == len(self.feature[1]) == len(self.feature[2])
        return len(self.feature[0])

    def __getitem__(self, idx):
        return tuple([el[idx] for el in self.feature])

    def _make_feature(self, raw_data):
        ids, masks, types, labels = [], [], [], []
        keyword_ids, adj_matrix = [], []

        for item_idx, item in enumerate(tqdm(raw_data)):
            if item_idx == 100:
                break
            context, response, negative1 = item
            pos_pair_keywords = get_keyword_from_utterance(context + " " + response)
            neg_pair_keywords = get_keyword_from_utterance(context + " " + negative1)

            positive = self.tokenizer(
                context,
                text_pair=response,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            negative1 = self.tokenizer(
                context,
                text_pair=negative1,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            ids.extend(positive["input_ids"])
            types.extend(positive["token_type_ids"])
            masks.extend(positive["attention_mask"])
            labels.append(0)
            ids.extend(negative1["input_ids"])
            types.extend(negative1["token_type_ids"])
            masks.extend(negative1["attention_mask"])
            labels.append(1)

            # Keyword
            random.shuffle(pos_pair_keywords)
            random.shuffle(neg_pair_keywords)
            pos_pair_keywords_ids = [
                0 if word not in self.c2i else self.c2i[word] for word in pos_pair_keywords
            ][: self.max_keyword_num]
            neg_pair_keywords_ids = [
                0 if word not in self.c2i else self.c2i[word] for word in neg_pair_keywords
            ][: self.max_keyword_num]

            if len(pos_pair_keywords_ids) < self.max_keyword_num:
                pos_pair_keywords_ids.extend(
                    [0] * (self.max_keyword_num - len(pos_pair_keywords_ids))
                )
            if len(neg_pair_keywords_ids) < self.max_keyword_num:
                neg_pair_keywords_ids.extend(
                    [0] * (self.max_keyword_num - len(neg_pair_keywords_ids))
                )
            # Keyword
            keyword_ids.append(pos_pair_keywords_ids)
            assert len(keyword_ids[-1]) == self.max_keyword_num
            keyword_ids.append(neg_pair_keywords_ids)
            assert len(keyword_ids[-1]) == self.max_keyword_num

            # Adjacent
            adj = np.zeros((self.max_keyword_num, self.max_keyword_num))
            for src_idx, src_id in enumerate(pos_pair_keywords_ids):
                for tgt_idx, tgt_id in enumerate(pos_pair_keywords_ids):
                    if src_id == 0 or tgt_id == 0:
                        adj[src_idx][tgt_idx] = -1
                    elif src_id not in self.cpnet.nodes() or tgt_id not in self.cpnet.nodes():
                        adj[src_idx][tgt_idx] = -1
                    else:
                        adj[src_idx][tgt_idx] = (
                            nx.shortest_path_length(self.cpnet, source=src_id, target=tgt_id) + 1
                        )
            adj_matrix.append(adj)
            adj = np.zeros((self.max_keyword_num, self.max_keyword_num))
            for src_idx, src_id in enumerate(neg_pair_keywords_ids):
                for tgt_idx, tgt_id in enumerate(neg_pair_keywords_ids):
                    if src_id == 0 or tgt_id == 0:
                        adj[src_idx][tgt_idx] = -1
                    elif src_id not in self.cpnet.nodes() or tgt_id not in self.cpnet.nodes():
                        adj[src_idx][tgt_idx] = -1
                    else:
                        adj[src_idx][tgt_idx] = (
                            nx.shortest_path_length(self.cpnet, source=src_id, target=tgt_id) + 1
                        )
            adj_matrix.append(adj)

        return (
            torch.stack(ids),
            torch.stack(types),
            torch.stack(masks),
            torch.tensor(labels),
            torch.tensor(keyword_ids),
            torch.tensor(adj_matrix),
        )


class GraphEvalDataset:
    def __init__(self, dataset, max_seq_len: int, tokenizer, cpnet, c2i, i2c, w2v, max_keyword_num):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.cpnet = cpnet
        self.c2i = c2i
        self.i2c = i2c
        self.max_keyword_num = max_keyword_num

        self.feature = self.make_feature(dataset)

    def make_feature(self, raw_zhao_data):
        """
        어차피 inference용이니깐 대충 만들어서 돌리기
        """
        encoded_list = []
        for item in raw_zhao_data:
            keyword_ids = []
            ctx = TURN_TOKEN.join(item["ctx"])
            # ref = item["ref"]
            hyp = item["hyp"]
            score = item["human_score"]
            encoded = self.tokenizer(
                ctx,
                text_pair=hyp,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            encoded["human_score"] = score

            # Keyword
            pos_pair_keywords = get_keyword_from_utterance(ctx + " " + hyp)
            random.shuffle(pos_pair_keywords)

            pos_pair_keywords_ids = [
                0 if word not in self.c2i else self.c2i[word] for word in pos_pair_keywords
            ][: self.max_keyword_num]

            if len(pos_pair_keywords_ids) < self.max_keyword_num:
                pos_pair_keywords_ids.extend(
                    [0 * (self.max_keyword_num - len(pos_pair_keywords_ids))]
                )
            # Keyword
            keyword_ids.append(pos_pair_keywords_ids)
            # Adjacent
            adj = np.zeros((self.max_keyword_num, self.max_keyword_num))
            for src_idx, src_id in enumerate(pos_pair_keywords_ids):
                for tgt_idx, tgt_id in enumerate(pos_pair_keywords_ids):
                    if src_id == 0 or tgt_id == 0:
                        adj[src_idx][tgt_idx] = -1
                    elif src_id not in self.cpnet.nodes() or tgt_id not in self.cpnet.nodes():
                        adj[src_idx][tgt_idx] = -1
                    else:
                        adj[src_idx][tgt_idx] = (
                            nx.shortest_path_length(self.cpnet, source=src_id, target=tgt_id) + 1
                        )

            encoded["keywords"] = keyword_ids
            encoded["adj"] = adj
            encoded_list.append(encoded)
        return encoded_list


class NSPDataset(Dataset):
    def __init__(
        self,
        fname,
        max_seq_len: int,
        tokenizer,
        num_neg: int = 1,
        rank_loss: bool = False,
    ):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.num_neg = num_neg
        self.fname = fname

        self.raw_data = self.read_dataset(fname)
        self.rank_loss = rank_loss
        if not self.rank_loss:
            self.feature = self._make_feature(self.raw_data)
        else:
            setname = "train" if "train" in self.fname else "valid"
            self.rand2_fname = "./data/negative/random_neg2_{}.txt".format(setname)
            self.random_neg2_dataset = self.read_dataset(self.rand2_fname)
            self.feature = self._make_feature_for_rank_loss(self.raw_data)

    def _get_random_negative(self, idx, setname):
        """
        혹시 negative가 없는데 ranking  loss를 쓰면 대신 random negative를 backup으로 씀.
        """
        assert setname in self.fname
        neg = self.random_neg2_dataset[idx][-1]
        assert isinstance(neg, str)
        return neg

    def _make_feature_for_rank_loss(self, raw_data):
        (
            context_ids,
            context_masks,
            golden_ids,
            golden_masks,
            negative1_ids,
            negative1_masks,
        ) = [[] for _ in range(6)]
        if self.num_neg == 2:
            negative2_ids, negative2_masks = [], []

        for item_idx, item in enumerate(tqdm(raw_data)):
            if item_idx == 100:
                break
            if self.num_neg == 1:
                context, response, negative1 = item
            elif self.num_neg == 2:
                context, response, negative1, negative2 = item
            else:
                raise ValueError()

            context = self.tokenizer(
                context,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            context_ids.extend(context["input_ids"])
            context_masks.extend(context["attention_mask"])
            response = self.tokenizer(
                response,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            golden_ids.extend(response["input_ids"])
            golden_masks.extend(response["attention_mask"])
            negative1 = self.tokenizer(
                negative1,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            negative1_ids.extend(negative1["input_ids"])
            negative1_masks.extend(negative1["attention_mask"])
            if self.num_neg == 2:
                if negative2 == "[NONE]":
                    negative2 = self._get_random_negative(
                        item_idx,
                        setname="train" if "train" in self.fname else "valid",
                    )
                negative2 = self.tokenizer(
                    negative2,
                    max_length=128,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                negative2_ids.extend(negative2["input_ids"])
                negative2_masks.extend(negative2["attention_mask"])

        if self.num_neg == 2:
            return (
                torch.stack(context_ids),
                torch.stack(context_masks),
                torch.stack(golden_ids),
                torch.stack(golden_masks),
                torch.stack(negative1_ids),
                torch.stack(negative1_masks),
                torch.stack(negative2_ids),
                torch.stack(negative2_masks),
            )
        else:
            return (
                torch.stack(context_ids),
                torch.stack(context_masks),
                torch.stack(golden_ids),
                torch.stack(golden_masks),
                torch.stack(negative1_ids),
                torch.stack(negative1_masks),
            )

    def read_dataset(self, fname):
        raw = read_raw_file(fname)
        removed = [el for el in raw if len(el) != self.num_neg + 2]
        for el in removed:
            raw.remove(el)
        assert all([len(el) == self.num_neg + 2 for el in raw])
        return raw

    def __len__(self):
        assert len(self.feature[0]) == len(self.feature[1]) == len(self.feature[2])
        return len(self.feature[0])

    def __getitem__(self, idx):
        return tuple([el[idx] for el in self.feature])

    def _make_feature(self, raw_data):
        ids, masks, types, labels = [], [], [], []

        for item_idx, item in enumerate(tqdm(raw_data)):
            if self.num_neg == 1:
                context, response, negative1 = item
            elif self.num_neg == 2:
                context, response, negative1, negative2 = item

            positive = self.tokenizer(
                context,
                text_pair=response,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            negative1 = self.tokenizer(
                context,
                text_pair=negative1,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            ids.extend(positive["input_ids"])
            types.extend(positive["token_type_ids"])
            masks.extend(positive["attention_mask"])
            labels.append(0)
            ids.extend(negative1["input_ids"])
            types.extend(negative1["token_type_ids"])
            masks.extend(negative1["attention_mask"])
            labels.append(1)
            if self.num_neg == 2 and negative2 != "[NONE]":
                negative2 = self.tokenizer(
                    context,
                    text_pair=negative2,
                    max_length=128,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )

                ids.extend(negative2["input_ids"])
                types.extend(negative2["token_type_ids"])
                masks.extend(negative2["attention_mask"])
                labels.append(1)

        return (
            torch.stack(ids),
            torch.stack(types),
            torch.stack(masks),
            torch.tensor(labels),
        )


class EvalDataset:
    def __init__(self, dataset, max_seq_len: int, tokenizer, rank_loss: bool = False):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        if rank_loss:
            self.feature = self.make_feature_for_rank(dataset)
        else:
            self.feature = self.make_feature(dataset)

    def make_feature_for_rank(self, raw_zhao_data):
        encoded_list = []
        for item in raw_zhao_data:
            ctx = TURN_TOKEN.join(item["ctx"])
            # ref = item["ref"]
            hyp = item["hyp"]
            score = item["human_score"]
            encoded = {}
            ctx = self.tokenizer(
                ctx,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            hyp = self.tokenizer(
                hyp,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            encoded["ctx"] = ctx
            encoded["hyp"] = hyp
            encoded["human_score"] = score
            encoded_list.append(encoded)
        return encoded_list

    def make_feature(self, raw_zhao_data):
        """
        어차피 inference용이니깐 대충 만들어서 돌리기
        """
        encoded_list = []
        for item in raw_zhao_data:
            ctx = TURN_TOKEN.join(item["ctx"])
            # ref = item["ref"]
            hyp = item["hyp"]
            score = item["human_score"]
            encoded = self.tokenizer(
                ctx,
                text_pair=hyp,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            encoded["human_score"] = score
            encoded_list.append(encoded)
        return encoded_list
