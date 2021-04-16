import argparse
import os

import torch
from tensorboardX import SummaryWriter
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm import tqdm
from transformers import BertConfig, BertModel, BertTokenizer

from get_dataset import get_dd_corpus
from utils import (dump_config, load_model, save_model, set_random_seed,
                   write_summary)

TURN_TOKEN = "[SEPT]"


class NSPDataset(Dataset):
    def __init__(self, raw_data, max_seq_len: int, tokenizer):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.feature = self._make_feature(raw_data)

    def __len__(self):
        return len(self.feature[0])

    def __getitem__(self, idx):
        return tuple([el[idx] for el in self.feature])

    def _make_feature(self, raw_data):
        ctx_inp_ids, ctx_masks, res_inp_ids, res_masks = [[] for _ in range(4)]

        for item_idx, item in enumerate(raw_data):
            if item_idx % 100 == 0:
                print(f"{item_idx}/{len(raw_data)}")
            assert isinstance(item, list) and all([isinstance(el, str) for el in item])

            if len(item) > 6:
                item = item[:6]

            for uttr_idx in range(len(item) - 1):
                uttrs = item[: uttr_idx + 2]
                assert len(uttrs) <= 6

                context, response = TURN_TOKEN.join(uttrs[:-1]), uttrs[-1]

                context = self.tokenizer(
                    context,
                    max_length=self.max_seq_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                response = self.tokenizer(
                    response,
                    max_length=self.max_seq_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                ctx_inp_ids.extend(context["input_ids"])
                ctx_masks.extend(context["attention_mask"])
                res_inp_ids.extend(response["input_ids"])
                res_masks.extend(response["attention_mask"])

        return (
            torch.stack(ctx_inp_ids),
            torch.stack(ctx_masks),
            torch.stack(res_inp_ids),
            torch.stack(res_masks),
        )


def make_annotated_dataset(raw_zhao_data, tokenizer):
    ctx_ids, ref_ids, hyp_ids, human_score = [], [], [], []
    for item in raw_zhao_data:
        ctx = "[SEPT]".join(item["ctx"])
        ref = item["ref"]
        hyp = item["hyp"]
        score = item["human_score"]
        ctx_ids.append(torch.tensor(tokenizer(ctx)["input_ids"][:128]))
        ref_ids.append(torch.tensor(tokenizer(ref)["input_ids"][:128]))
        hyp_ids.append(torch.tensor(tokenizer(hyp)["input_ids"][:128]))
        human_score.append(score)
    return ctx_ids, ref_ids, hyp_ids, human_score


def main(args):
    set_random_seed(42)
    dump_config(args)
    device = torch.device("cuda")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    special_tokens_dict = {"additional_special_tokens": ["[SEPT]"]}
    tokenizer.add_special_tokens(special_tokens_dict)

    context_encoder, response_encoder = (
        BertModel.from_pretrained("bert-base-uncased"),
        BertModel.from_pretrained("bert-base-uncased"),
    )
    context_encoder.resize_token_embeddings(len(tokenizer))
    response_encoder.resize_token_embeddings(len(tokenizer))
    context_encoder.to(device)
    response_encoder.to(device)

    raw_dd_train, raw_dd_valid = get_dd_corpus("train"), get_dd_corpus("validation")
    train_dataset, valid_dataset = (
        NSPDataset(raw_dd_train, 128, tokenizer),
        NSPDataset(raw_dd_valid, 128, tokenizer),
    )

    trainloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=args.batch_size,
        drop_last=True,
    )
    validloader = DataLoader(valid_dataset, batch_size=args.batch_size, drop_last=True)

    optimizer = AdamW(
        list(context_encoder.parameters()) + list(response_encoder.parameters()),
        lr=args.lr,
    )

    total_step = args.epoch * len(trainloader)
    writer = SummaryWriter(args.board_path)
    global_step = 0
    print("GOGO~")
    criteria = torch.nn.CrossEntropyLoss()
    label = torch.tensor([_ for _ in range(args.batch_size)]).to(device)
    for epoch in range(args.epoch):
        print("Epoch {}".format(epoch))

        context_encoder.train()
        response_encoder.train()
        for step, batch in enumerate(tqdm(trainloader)):
            global_step += 1
            ctx_ids, ctx_mask, res_ids, res_mask = [el.to(device) for el in batch]

            ctx_encoded = context_encoder(ctx_ids, ctx_mask, return_dict=True)[
                "pooler_output"
            ]
            res_encoded = response_encoder(res_ids, res_mask, return_dict=True)[
                "pooler_output"
            ]
            output = torch.matmul(ctx_encoded, res_encoded.T)
            response_select_loss = criteria(output, label)
            context_select_loss = criteria(
                torch.matmul(res_encoded, ctx_encoded.T), label
            )
            for p in context_encoder.pooler.dense.parameters():
                context_l2_loss = 0.01 * torch.norm(p)
            for p in response_encoder.pooler.dense.parameters():
                response_l2_loss = 0.01 * torch.norm(p)
            loss = (
                response_select_loss
                + context_select_loss
                + context_l2_loss
                + response_l2_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(context_encoder.parameters())
                + list(response_encoder.parameters()),
                5.0,
            )
            optimizer.step()
            optimizer.zero_grad()
            write_summary(
                writer,
                {"r_select": response_select_loss},
                "train",
                global_step,
            )
            write_summary(
                writer,
                {"c_select": context_select_loss},
                "train",
                global_step,
            )
            write_summary(writer, {"context_l2": context_l2_loss}, "train", global_step)
            write_summary(
                writer,
                {"response_l2": response_l2_loss},
                "train",
                global_step,
            )
            write_summary(writer, {"loss": loss}, "train", global_step)

        context_encoder.eval()
        response_encoder.eval()

        with torch.no_grad():
            loss_list = []
            for step, batch in enumerate(tqdm(validloader)):
                ctx_ids, ctx_mask, res_ids, res_mask = [el.to(device) for el in batch]

                ctx_encoded = context_encoder(ctx_ids, ctx_mask, return_dict=True)[
                    "pooler_output"
                ]
                res_encoded = response_encoder(res_ids, res_mask, return_dict=True)[
                    "pooler_output"
                ]
                output = torch.matmul(ctx_encoded, res_encoded.T)
                response_loss = criteria(output, label)
                context_loss = criteria(torch.matmul(res_encoded, ctx_encoded.T), label)
                loss = response_loss + context_loss
                loss_list.append(loss.cpu().detach().numpy())
            final_loss = sum(loss_list) / len(loss_list)
            write_summary(writer, {"loss": loss}, "valid", global_step)

        save_model(context_encoder, args.context_model_path, epoch)
        save_model(response_encoder, args.response_model_path, epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--exp_name", type=str, default="test")
    parser.add_argument("--log_path", type=str, default="logs")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epoch", type=int, default=10)

    args = parser.parse_args()

    args.exp_path = os.path.join(args.log_path, args.exp_name)
    args.context_model_path = os.path.join(args.exp_path, "context_model")
    args.response_model_path = os.path.join(args.exp_path, "response_model")
    args.board_path = os.path.join(args.exp_path, "board")
    os.makedirs(args.context_model_path, exist_ok=True)
    os.makedirs(args.response_model_path, exist_ok=True)
    os.makedirs(args.board_path, exist_ok=True)
    main(args)
