import os
from pathlib import Path
from enum import StrEnum
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer
from rich.progress import track

from bert_tutorial.data_loader import generate_q_and_a_pairs

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
TOKENIZER_DIR = ROOT_DIR / "bert-it-1"

MAX_CONVERSATION_PAIRS = 100


class SpecialToken(StrEnum):
    PAD = "[PAD]" # 文頭
    SEP = "[SEP]" # 文末
    UNK = "[UNK]"
    CLS = "[CLS]"
    MASK = "[MASK]"

    @staticmethod
    def to_list():
        return [token.value for token in SpecialToken]


def _prepare_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if not os.path.exists(TOKENIZER_DIR):
        os.makedirs(TOKENIZER_DIR)


def _save_conversations(conversation_pairs, overwrite=False):
    if not overwrite:
        if DATA_DIR.exists() and DATA_DIR.glob("**/*.txt"):
            raise FileExistsError("Conversation files already exist")
    text_data = []
    file_num = len(conversation_pairs) // MAX_CONVERSATION_PAIRS
    for file_count in track(range(file_num), description="Saving conversations"):
        for _ in range(MAX_CONVERSATION_PAIRS):
            text_data.append(conversation_pairs.pop(0)[0])
        with open(DATA_DIR / f"text_{file_count+1}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(text_data))
        text_data.clear()


def _data_paths():
    paths = [str(path) for path in DATA_DIR.glob("**/*.txt")]
    return paths


def train_tokenizer(paths):
    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=True,
    )
    tokenizer.train(
        files=paths,
        vocab_size=30_000,
        min_frequency=5,
        limit_alphabet=1000,
        wordpieces_prefix="##",
        special_tokens=SpecialToken.to_list(),
    )

    tokenizer.save_model(str(TOKENIZER_DIR), "bert-it")
    tokenizer = BertTokenizer.from_pretrained(
        str(TOKENIZER_DIR / "bert-it-vocab.txt"), local_files_only=True
    )


if __name__ == "__main__":
    pairs = generate_q_and_a_pairs()
    _prepare_data_dir()
    try:
        _save_conversations(pairs)
    except FileExistsError as e:
        pass
    conversation_paths = _data_paths()
    train_tokenizer(conversation_paths)
