import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
TURN_TOKEN = "[SEPT]"


class NSPDataset(Dataset):
    def __init__(self, fname, max_seq_len: int, tokenizer, num_neg: int = 1):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.num_neg = num_neg
        raw_data = self.read_dataset(fname)
        self.feature = self._make_feature(raw_data)

    def read_dataset(self, fname):
        with open(fname, "r") as f:
            ls = [el.strip().split("|||") for el in f.readlines()]
        assert all([len(el) == self.num_neg + 2 for el in ls])
        return ls

    def __len__(self):
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
    def __init__(self, dataset, max_seq_len: int, tokenizer):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.feature = self.make_feature(dataset)

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