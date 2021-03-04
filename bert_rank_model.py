from torch import nn
import torch


class BertRankModel(torch.nn.Module):
    def __init__(self, bert, bert_config, config):
        super(BertRankModel, self).__init__()

        self.bert = bert
        for param in self.bert.parameters():
            param.requires_grad = False
        self.config = config
        self.bert_config = bert_config

        self.M = nn.Parameter(
            torch.FloatTensor(
                self.bert_config.hidden_size, self.bert_config.hidden_size
            )
        )
        self.prediction_layer = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(
                2 * self.bert_config.hidden_size + 1,
                self.bert_config.hidden_size,
            ),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(self.bert_config.hidden_size, 1),
            torch.sigmoid(),
        )

    def forward(self, context_ids, context_mask, response_ids, response_mask):
        context_outputs, _ = self.bert(
            context_ids, attention_mask=context_mask
        )
        response_outputs, _ = self.bert(
            response_ids, attention_mask=response_mask
        )

        expanded_context_mask = (
            context_mask.unsqueeze(-1).expand(context_outputs.size()).float()
        )
        expanded_response_mask = (
            response_mask.unsqueeze(-1)
            .expand(response_outputs.size())
            .float()
        )
        expanded_context_mask[expanded_context_mask == 0] = -1e9
        context_outputs = context_outputs * expanded_context_mask
        pooled_context, _ = torch.max(context_outputs, 1)

        expanded_response_mask[expanded_response_mask == 0] = -1e9
        response_outputs = response_outputs * expanded_response_mask
        pooled_response, _ = torch.max(response_outputs, 1)

        context_response_dot = (
            (
                torch.matmul(pooled_context, self.attention_matrix)
                * pooled_response
            )
            .sum(1)
            .view(-1, 1)
        )

        response_concat = torch.cat(
            [pooled_response, context_response_dot, pooled_context], 1
        )

        response_prob = self.prediction_layer(response_concat)
        return response_prob
