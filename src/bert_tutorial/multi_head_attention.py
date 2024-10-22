import torch
from torch import nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, heads: int, d_model: int, dropout=0.1):
        super().__init__()

        assert d_model % heads == 0
        self.d_k = d_model // heads  #
        self.heads = heads
        self.dropout = nn.Dropout(dropout)

        self.query = nn.Linear(in_features=d_model, out_features=d_model)
        self.key = nn.Linear(in_features=d_model, out_features=d_model)
        self.value = nn.Linear(in_features=d_model, out_features=d_model)
        self.output_liner = nn.Linear(in_features=d_model, out_features=d_model)

    def forwad(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
    ):
        """

        Args:
            query (torch.Tensor): (batch_size, max_len, d_model)
            key (torch.Tensor): (batch_size, max_len, d_model)
            value (torch.Tensor): (batch_size, max_len, d_model)
            mask (torch.Tensor): (batch_size, 1, 1, max_words)

        """
        # (batch_size, max_len, d_model)
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # (batch_size, max_len, d_model) --> (batch_size, max_len, h, d_k) --> (batch_size, h, max_len, d_k)
        # torch.Tensor.view()は引数に-1を渡すとそこは自動で調整してくれる
        # query.shape[0] = batch_size
        # # torch.Tensor.permute() 並び替える順番をタプル型で渡す。
        query = query.view(query.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)
        key = key.view(key.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)
        value = value.view(value.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)

        # (batch_size, h, max_len, d_k) matmul (batch_size, h, d_k, max_len) --> (batch_size, h, max_len, max_len)
        scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / math.sqrt(
            query.size(-1)  # d_k
        )
        # ソフトマックスのウェイトに影響を与えないように、0マスクを超小さい数字で埋める。
        # (batch_size, h, max_len, max_len)
        scores = scores.masked_fill(mask == 0, -1e9)

        # (batch_size, h, max_len, max_len)
        # パッド以外のすべてのトークンにAttentionの重みを置くソフトマックス
        # max_len X max_len matrix of attention
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # (batch_size, h, max_len, max_len) matmul (batch_size, h, max_len, d_k) --> (batch_size, h, max_len, d_k)
        context = torch.matmul(weights, value)

        # 下の形に戻す？
        # (batch_size, h, max_len, d_k) --> (batch_size, max_len, h, d_k) --> (batch_size, max_len, d_model)
        context = (
            context.permute(0, 2, 1, 3)
            .contiguous()
            .view(context.shape[0], -1, self.heads * self.d_k)
        )

        # (batch_size, max_len, d_model)
        return self.output_linear(context)
