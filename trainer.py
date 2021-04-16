import os
from logging import Logger
from typing import Dict, Tuple, Union

import numpy as np
import scipy
import torch
import torch.nn as nn
import transformers
from tensorboardX import SummaryWriter
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from utils import eval_by_NSP


class Trainer:
    def __init__(
        self,
        config,
        model,
        train_loader,
        valid_loader,
        logger: Logger,
        device,
        dataset_for_correlation,
        is_rank_loss: bool = False,
    ):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.logger = logger
        self.device = device
        self.config = config
        self.is_rank_loss = is_rank_loss
        self.crossentropy = CrossEntropyLoss()
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.lr,
        )

        self.steps_per_epoch = len(self.train_loader)
        self.total_steps = self.steps_per_epoch * self.config.epoch
        self.eval_dataset_for_correlation = dataset_for_correlation

        self.writer = SummaryWriter(self.config.board_path)

    def train(self):
        self.logger.info("=== training ===")
        global_step = 0
        os.makedirs(self.config.model_path, exist_ok=True)
        os.makedirs(self.config.board_path, exist_ok=True)

        self.model.eval()
        # self._calc_correlation(global_step)
        # val_loss = self._validation(global_step)
        self.model.train()

        for epoch in range(self.config.epoch):
            self.logger.info("epoch {}".format(epoch))
            for step, batch in enumerate(tqdm(self.train_loader), start=1):
                global_step += 1
                batch = tuple(el.to(self.device) for el in batch)
                loss, perf = self._train_step(batch)
                perf["loss"] = loss
                self._write_summary(
                    perf,
                    "TRAIN",
                    global_step,
                )
            self.model.eval()
            val_loss = self._validation(global_step)
            self._calc_correlation(global_step)

            self.model.train()
            self._save_model(epoch)

    def _calc_correlation(self, global_step):
        evaluation_result = eval_by_NSP(
            self.eval_dataset_for_correlation,
            self.model,
            self.device,
            is_rank=self.is_rank_loss,
        )

        result = {}
        for item in evaluation_result:
            for k, v in item.items():
                if k not in result:
                    result[k] = []
                result[k].append(v)

        pearson, spearman = {}, {}
        for k, v in result.items():
            if k == "human_score":
                continue
            # try:
            pearson_result = scipy.stats.pearsonr(result["human_score"], v)
            spearman_result = scipy.stats.spearmanr(result["human_score"], v)
            # except:
            #    pearson_result = [-11, -11]
            #    spearman_result = [-11, -11]

            pearson["pearson/" + k] = pearson_result[0]
            pearson["pearson_p/" + k] = pearson_result[1]
            spearman["spearman/" + k] = spearman_result[0]
            spearman["spearman_p/" + k] = spearman_result[1]
        self._write_summary(pearson, "VALID", global_step)
        self._write_summary(spearman, "VALID", global_step)

    def _validation(self, global_step: int):
        loss_list = []

        with torch.no_grad():
            for step, batch in enumerate(tqdm(self.valid_loader), start=1):
                batch = tuple(el.to(self.device) for el in batch)
                if self.is_rank_loss:
                    if self.config.num_neg == 1:
                        (
                            ctx_ids,
                            ctx_mask,
                            golden_ids,
                            golden_mask,
                            neg1_ids,
                            neg1_mask,
                        ) = batch
                    if self.config.num_neg == 2:
                        (
                            ctx_ids,
                            ctx_mask,
                            golden_ids,
                            golden_mask,
                            neg1_ids,
                            neg1_mask,
                            neg2_ids,
                            neg2_mask,
                        ) = batch
                    # shape of [batch_size, 1]
                    golden_score = self.model(
                        ctx_ids, ctx_mask, golden_ids, golden_mask
                    )
                    neg1_score = self.model(ctx_ids, ctx_mask, neg1_ids, neg1_mask)
                    if self.config.num_neg == 2:
                        neg2_score = self.model(ctx_ids, ctx_mask, neg2_ids, neg2_mask)
                    prediction = torch.cat([golden_score, neg1_score, neg2_score], 1)
                    loss = self.crossentropy(
                        prediction,
                        torch.zeros(self.config.batch_size, dtype=torch.long).to(
                            self.device
                        ),
                    )
                else:
                    ids, token_type, attn, label = batch
                    output = self.model(
                        # ids, attn, token_type, next_sentence_label=label
                        ids,
                        attn,
                        next_sentence_label=label,
                    )
                    loss = output[0]

                loss_list.append(loss.cpu().detach().numpy())

        final_loss = sum(loss_list) / len(loss_list)
        self._write_summary({"loss": final_loss}, "VALID", global_step)
        return final_loss

    def _train_step(self, batch):
        if self.is_rank_loss:
            if self.config.num_neg == 1:
                (
                    ctx_ids,
                    ctx_mask,
                    golden_ids,
                    golden_mask,
                    neg1_ids,
                    neg1_mask,
                ) = batch
            if self.config.num_neg == 2:
                (
                    ctx_ids,
                    ctx_mask,
                    golden_ids,
                    golden_mask,
                    neg1_ids,
                    neg1_mask,
                    neg2_ids,
                    neg2_mask,
                ) = batch
            # shape of [batch_size, 1]
            golden_score = self.model(ctx_ids, ctx_mask, golden_ids, golden_mask)
            neg1_score = self.model(ctx_ids, ctx_mask, neg1_ids, neg1_mask)
            if self.config.num_neg == 2:
                neg2_score = self.model(ctx_ids, ctx_mask, neg2_ids, neg2_mask)

            prediction = torch.cat([golden_score, neg1_score, neg2_score], 1)

            loss = self.crossentropy(
                prediction,
                torch.zeros(self.config.batch_size, dtype=torch.long).to(self.device),
            )
        else:
            ids, token_type, attn, label = batch
            output = self.model(
                # ids, attn, token_type, next_sentence_label=label
                ids,
                attn,
                next_sentence_label=label,
            )
            loss = output[0]
        perf = {}
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss, perf

    def _write_summary(self, values: Dict[str, torch.Tensor], setname: str, step: int):
        for k, v in values.items():
            self.writer.add_scalars(k, {setname: v}, step)
        self.writer.flush()

    def _save_model(self, epoch: int):
        torch.save(
            self.model.state_dict(),
            os.path.join(self.config.model_path, f"epoch-{epoch}.pth"),
        )
