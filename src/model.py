from torch import nn
from bert_tutorial import BERTEmbedding, EncoderLayer


class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, d_model=768, n_layers=12, heads=12, dropout=0.1):
        """BERTモデル

        Args:
            vocab_size (int): 総語彙数
            d_model (int, optional): 隠れ層のサイズ. Defaults to 768.
            n_layers (int, optional): トランスフォーマーブロックの数. Defaults to 12.
            heads (int, optional): Attentionのヘッド数. Defaults to 12.
            dropout (float, optional): ドロップアウトの割合. Defaults to 0.1.
        """
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.heads = heads

        # paper noted they used 4 * hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = d_model * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=d_model)

        # multi-layers transformer blocks, deep network
        self.encoder_blocks = nn.ModuleList(
            [
                EncoderLayer(d_model, heads, d_model * 4, dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(self, x, segment_info):
        # PADトークンへのマスキング
        # (batch_size, 1, seq_len, seq_len)
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # インデックス付きシーケンスをベクトルシーケンスに埋め込む
        x = self.embedding(x, segment_info)

        # 複数のトランス・ブロックにまたがる
        for encoder in self.encoder_blocks:
            x = encoder.forward(x, mask)
        return x


class NextSentencePrediction(nn.Module):
    """
    2-class classification model : is_next, is_not_next
    """

    def __init__(self, hidden):
        """

        Args:
            hidden (int): BERTのモデルの出力サイズ
        """
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # 最初の [CLS]トークンのみを用いる
        return self.softmax(self.linear(x[:, 0]))


class MaskedLanguageModel(nn.Module):
    """
    マスクされた入力シーケンスから下のトークンを予測する nクラス分類問題
    n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        Args:
            hidden: BERTの出力サイズ
            vocab_size: 総語彙数
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))


class BERTLM(nn.Module):
    """
    BERT Language Models
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(self, bert: BERT, vocab_size):
        """
        Args:
            bert: BERT model which should be trained
            vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        self.next_sentence = NextSentencePrediction(self.bert.d_model)
        self.mask_lm = MaskedLanguageModel(self.bert.d_model, vocab_size)

    def forward(self, x, segment_label):
        x = self.bert(x, segment_label)
        return self.next_sentence(x), self.mask_lm(x)
