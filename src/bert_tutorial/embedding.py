import math
import torch


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()

        # 対数空間でPositional Encodingを行う
        pe: torch.Tensor = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        for pos in range(max_len):
            # それぞれの位置のそれぞれの次元について
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** (2 * (i + 1) / d_model)))
        # バッチサイズを含める
        self.pe = pe.unsqueeze(0)  # unsqueeze: 指定した位置にサイズ1の次元を挿入する

    def forward(self, x: torch.Tensor):
        self.pe = self.pe.to(x.device)
        assert (
            x.device == self.pe.device
        ), "入力テンソルと positional encoding テンソルは同じデバイスにある必要があります"
        return self.pe


class BERTEmbedding(torch.nn.Module):
    """BERTのEmbeddingを行う

    BERTのEmbeddingは以下の要素で構成されている
    1. TokenEmbedding: 普通の埋め込み行列
    2. PositionalEmbedding: sin/cosを使って位置情報を埋め込む
    3. SegmentEmbedding: 文のセグメント情報を追加する。(sent_A:1, sent_B:2)


    Args:

    """

    def __init__(self, vocab_size: int, embed_size: int, seq_len=64, dropout=0.1):
        """_summary_

        Args:
            vocab_size (int): 語彙の総数
            embed_size (int): Token Embeddingの埋め込みサイズ
            seq_len (int, optional): シーケンス長. Defaults to 64.
            dropout (float, optional): Dropoutの割合. Defaults to 0.1.
        """
        super().__init__()
        self.embed_size = embed_size
        # (m, seq_len) --> (m, seq_len, embed_size)
        # padding_idxは訓練中は更新されず、pad(0)のままで固定される
        self.token = torch.nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.segment = torch.nn.Embedding(3, embed_size, padding_idx=0)
        self.position = PositionalEmbedding(d_model=embed_size, max_len=seq_len)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, sequence: torch.Tensor, segment_label: torch.Tensor):
        x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        return self.dropout(x)
