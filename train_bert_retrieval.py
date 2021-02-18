import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from transformers import (
    BertTokenizer,
    BertForNextSentencePrediction,
    BertConfig,
)
from utils import (
    set_random_seed,
    dump_config,
    save_model,
    load_model,
    write_summary,
)
from get_dataset import get_dd_corpus, get_zhao_dataset
import argparse
import os
from torch.optim.adamw import AdamW
from tensorboardX import SummaryWriter
from tqdm import tqdm
from trainer import Trainer
from utils import get_logger

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


def main(args):
    set_random_seed(42)
    logger = get_logger()

    dump_config(args)
    device = torch.device("cuda")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    special_tokens_dict = {"additional_special_tokens": ["[SEPT]"]}
    tokenizer.add_special_tokens(special_tokens_dict)

    model = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    dataset_for_correlation = EvalDataset(
        get_zhao_dataset("dd"),
        128,
        tokenizer,
    )

    train_dataset, valid_dataset = (
        NSPDataset(
            args.data_path.format(args.num_neg, "train"),
            128,
            tokenizer,
            num_neg=args.num_neg,
        ),
        NSPDataset(
            args.data_path.format(args.num_neg, "valid"),
            128,
            tokenizer,
            num_neg=args.num_neg,
        ),
    )
    trainloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=args.batch_size,
        drop_last=True,
    )
    validloader = DataLoader(
        valid_dataset, batch_size=args.batch_size, drop_last=True
    )

    trainer = Trainer(
        args,
        model,
        trainloader,
        validloader,
        logger,
        device,
        dataset_for_correlation,
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--exp_name",
        type=str,
        default="del_prev_turn-topk100_neg2",  # "rand_neg1"
    )  # "prefix-topk100_neg2")
    parser.add_argument("--num_neg", type=int, default=2)
    parser.add_argument("--log_path", type=str, default="logs")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/negative/del_prev_turn-topk100_neg{}_{}.txt",  # "./data/negative/prefix-topk100_neg{}_{}.txt",
    )

    args = parser.parse_args()

    args.exp_path = os.path.join(args.log_path, args.exp_name)
    args.model_path = os.path.join(args.exp_path, "model")
    args.board_path = os.path.join(args.exp_path, "board")
    from pprint import pprint

    pprint(args)
    input(">> RIGHT?")
    os.makedirs(args.model_path, exist_ok=False)
    os.makedirs(args.board_path, exist_ok=False)
    main(args)
