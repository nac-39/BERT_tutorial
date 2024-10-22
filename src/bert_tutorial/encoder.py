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
        self.output_linear = nn.Linear(in_features=d_model, out_features=d_model)

    def forward(
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


class FeedForwad(nn.Module):
    def __init__(self, d_model: int, middle_dim: int = 2048, drouout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, middle_dim)
        self.fc2 = nn.Linear(middle_dim, d_model)
        self.dropout = nn.Dropout(drouout)
        self.activation = nn.GELU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, d_model=768, heads=12, feed_forward_hidden=768 * 4, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.self_multihead = MultiHeadAttention(heads, d_model)
        self.feed_forward = FeedForwad(d_model, middle_dim=feed_forward_hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings, mask):
        # embeddings: (batch_size, max_len, d_model)
        # encoder mask: (batch_size, 1, 1, max_len)
        # result: (batch_size, max_len, d_model)
        interacted = self.dropout(
            self.self_multihead(embeddings, embeddings, embeddings, mask)
        )
        # 残差レイヤー？勾配消失しないようにする。
        interacted = self.layer_norm(interacted + embeddings)
        # ボトルネック
        feed_forward_out = self.dropout(self.feed_forward(interacted))
        encoded = self.layer_norm(feed_forward_out + interacted)
        return encoded
