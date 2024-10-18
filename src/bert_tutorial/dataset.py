from torch.utils.data import Dataset
import random
import itertools


class BERTDataset(Dataset):
    def __init__(self, data_pair: tuple[str, str], tokenizer, seq_len: int = 64):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.corpus_lines = len(data_pair)
        self.lines = data_pair

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):

        # ステップ1: ネガティブなものもポジティブなものも含めて、ランダムな文章のペアを選択
        text1, text2, is_next_label = self.get_sent(item)
        # ステップ2: 文章の中のランダムな単語をMASKかランダムな単語に置き換える

    def random_word(self, sentence):
        tokens = sentence.split()
        output_label = []
        output = []

        # 15%のトークンを置き換える
        for i, token in enumerate(tokens):
            prob = random.random()
            # [CLS], [SEP]トークンを取り除く
            token_id = self.tokenizer(token)["input_ids"][1:-1]

            if prob < 0.15:
                prob /= 0.15
                if prob < 0.8:
                    # 80%の確率で[MASK]に置き換える
                    for _ in range(len(token_id)):
                        output.append(self.tokenizer.vocab("[MASK]"))

                # 10%の確率でランダムな単語に置き換える
                elif prob < 0.9:
                    for _ in range(len(token_id)):
                        output.append(random.randrange(len(self.tokenizer.vocab)))
                # 10%の確率でそのままにする
                else:
                    output.append(token_id)
                output_label.append(token_id)
            else:
                output.append(token_id)
                output_label.append([0] * len(token_id))
                # flattening
        output = list(
            itertools.chain(*[[x] if not isinstance(x, list) else x for x in output])
        )
        output_label = list(
            itertools.chain(
                *[[x] if not isinstance(x, list) else x for x in output_label]
            )
        )
        assert len(output) == len(output_label)
        return output, output_label

    def get_corpus_line(self, index: int):
        return self.lines[index]

    def get_sent(self, index: int):
        """ランダムな文章のペアを選択する"""
        text1, text2 = self.get_corpus_line(index)
        if random.choice([True, False]):
            is_next_label = 1
            return text1, text2, is_next_label
        else:
            random_text = self.get_random_line(index)
            is_next_label = 0
            return text1, random_text, is_next_label
