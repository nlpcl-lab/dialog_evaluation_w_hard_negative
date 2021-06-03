import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertModel


class GRADE(nn.Module):
    """model code"""

    def __init__(self, embedding_init_value, tokenizer_len):
        super().__init__()
        self.bert_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.bert_encoder.resize_token_embeddings(tokenizer_len)
        self.linear_onehop_weight = torch.nn.Linear(300 * (8 + 2), 300)
        self.linear_twohop_weight = torch.nn.Linear(300 * (8 + 2), 300)

        self.linear0_1 = torch.nn.Linear(300 * (8 + 2), 300)
        self.linear0_2 = torch.nn.Linear(300 * (8 + 2), 300)
        self.linear0_3 = torch.nn.Linear(300 * (8 + 2), 300)

        self.linear1 = torch.nn.Linear(300, 512)
        self.linear2_1 = torch.nn.Linear(768, 512)
        self.linear2_2 = torch.nn.Linear(768, 512)
        self.linear3 = torch.nn.Linear(1024, 512)
        self.linear4_1 = torch.nn.Linear(512, 128)
        self.linear4_2 = torch.nn.Linear(512, 128)
        self.linear5 = torch.nn.Linear(128, 1)

        self.word_embedder = nn.Embedding(num_embeddings=50000, embedding_dim=300, padding_idx=0)
        self.word_embedder.weight = nn.Parameter(torch.tensor(embedding_init_value))
        self.gat_1 = GATLayer(
            in_features=300,
            out_features=300,
            alpha=0.2,
            nheads=4,
            activation=False,
        )
        self.gat_2 = GATLayer(
            in_features=300,
            out_features=300,
            alpha=0.2,
            nheads=4,
            activation=False,
        )
        self.gat_3 = GATLayer(
            in_features=300,
            out_features=300,
            alpha=0.2,
            nheads=4,
            activation=False,
        )

    def forward(self, input_ids, input_masks, keyword_ids, adjs):
        keyword_h_embed = self.word_embedder(keyword_ids).float()
        keyword_nonzero = torch.count_nonzero(keyword_ids, dim=1)
        bs = input_ids.size(0)

        # gat_1
        keyword_z_embed = self.gat_1(keyword_h_embed, adjs)
        keyword_h_embed = F.elu(
            self.linear0_1(keyword_h_embed.reshape(bs, -1)).reshape(bs, -1, 300) + keyword_z_embed
        )
        # gat_2
        keyword_z_embed = self.gat_2(keyword_h_embed, adjs)
        keyword_h_embed = F.elu(
            self.linear0_2(keyword_h_embed.reshape(bs, -1)).reshape(bs, -1, 300) + keyword_z_embed
        )
        # gat_3
        keyword_z_embed = self.gat_3(keyword_h_embed, adjs)
        keyword_h_embed = F.elu(
            self.linear0_3(keyword_h_embed.reshape(bs, -1)).reshape(bs, -1, 300) + keyword_z_embed
        )
        # mean pool
        keyword_h_embed = F.elu(
            self.linear1(torch.div(torch.sum(keyword_h_embed, dim=1), keyword_nonzero.unsqueeze(1)))
        )

        bert_embed = self.bert_encoder(input_ids, input_masks, return_dict=True)[
            "last_hidden_state"
        ][:, 0]

        bert_embed = self.linear2_1(bert_embed)

        fusion_embs = torch.cat((bert_embed, keyword_h_embed), 1)
        fusion_embs = F.elu(self.linear3(fusion_embs))

        linear = F.elu(self.linear4_1(fusion_embs))
        linear = self.linear5(linear)
        score = torch.sigmoid(linear).squeeze()
        return score


class GATHead(nn.Module):
    def __init__(self, in_features, out_features, alpha, activation=True, device="cpu"):
        super(GATHead, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, input, adj):
        adj = adj.type(torch.FloatTensor)
        h = self.W(input)
        (B, N, _) = adj.shape
        a_input = torch.cat([h.repeat(1, 1, N).view(B, N * N, -1), h.repeat(1, N, 1)], dim=1).view(
            B, N, -1, 2 * self.out_features
        )
        e = self.leakyrelu(self.a(a_input).squeeze(3))

        zero_vec = -9e15 * torch.ones_like(e)

        attention = torch.where(adj > 0, e, zero_vec)
        attention = attention * adj
        attention = F.softmax(attention, dim=1)
        h_prime = torch.matmul(attention, h)

        if self.activation:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class GATLayer(nn.Module):
    """
    Graph Attention Layer, GAT paper at https://arxiv.org/abs/1710.10903
    Implementation inspired by https://github.com/Diego999/pyGAT
    """

    def __init__(self, in_features, out_features, alpha, nheads=1, activation=True, device="cpu"):
        """
        :param in_features:     size of the input per node
        :param out_features:    size of the output per node
        :param alpha:           slope of the leaky relu
        :param nheads:          number of attention heads
        :param activation:      whether to apply a non-linearity
        :param device:          device used for computation
        """
        super(GATLayer, self).__init__()
        assert out_features % nheads == 0

        self.input_head = in_features
        self.output_head = out_features // nheads

        self.heads = nn.ModuleList()
        for _ in range(nheads):
            self.heads.append(
                GATHead(
                    in_features=self.input_head,
                    out_features=self.output_head,
                    alpha=alpha,
                    activation=activation,
                )
            )

    def forward(self, input, adj):
        y = torch.cat([head(input, adj) for head in self.heads], dim=2)
        return y

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )
