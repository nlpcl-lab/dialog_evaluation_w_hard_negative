from torch import nn
import torch


class BertRankModel(torch.nn.Module):
    def __init__(self, bert, bert_config, config):
        super(BertRankModel, self).__init__()

        self.bert = bert

        self.config = config
        self.bert_config = bert_config

        self.M = nn.Parameter(
            torch.FloatTensor(
                self.bert_config.hidden_size, self.bert_config.hidden_size
            ), requires_grad=True
        )
        self.prediction_layer = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(
                2 * self.bert_config.hidden_size + 1,
                self.bert_config.hidden_size,
            ),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.bert_config.hidden_size, 1),
            nn.Tanh()
        )

        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
        self.prediction_layer.apply(init_weights)

    def forward(self, context_ids, context_mask, response_ids, response_mask):
        context_outputs, _ = self.bert(
            context_ids, attention_mask=context_mask, return_dict=False
        )
        response_outputs, _ = self.bert(
            response_ids, attention_mask=response_mask, return_dict=False
        )

        expanded_context_mask = (
            context_mask.unsqueeze(-1).expand(context_outputs.size()).float()
        )
        expanded_response_mask = (
            response_mask.unsqueeze(-1)
            .expand(response_outputs.size())
            .float()
        )
        pooled_context = torch.sum(
            context_outputs * expanded_context_mask, 1
        ) / torch.clamp(expanded_context_mask.sum(1), min=1e-9)
        pooled_response = torch.sum(
            response_outputs * expanded_response_mask, 1
        ) / torch.clamp(expanded_response_mask.sum(1), min=1e-9)

        context_response_dot = (
            (
                torch.matmul(pooled_context, self.M)
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
