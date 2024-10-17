from pathlib import Path
from enum import IntEnum

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MAX_LEN = 64  # 一文の最大トークン数は64
CORPUS_MOVIE_CONVERSATION = PROJECT_ROOT / "datasets" / "movie_conversations.txt"
CORPUS_MOVIE_LINES = PROJECT_ROOT / "datasets" / "movie_lines.txt"


class LinesIdx(IntEnum):
    LINE_ID = 0
    CONVERSATION = 4


class ConversationsIdx(IntEnum):
    CHARACTER1 = 0
    CHARACTER2 = 1
    MOVIE_ID = 2
    CONVERSATION_LIST = 3


def _load_datasets() -> tuple[str, str]:
    with open(CORPUS_MOVIE_CONVERSATION, "r", encoding="iso-8859-1") as f:
        conversations = f.readlines()

    with open(CORPUS_MOVIE_LINES, "r", encoding="iso-8859-1") as f:
        lines = f.readlines()

    return conversations, lines


def _split_lines_by_marker(lines: list[str]) -> dict[str, str]:
    """
    raw:
    L197 +++$+++ u2 +++$+++ m0 +++$+++ CAMERON +++$+++ Okay... then how 'bout we try out some French cuisine.  Saturday?  Night?
    → {L197: "Okay... then how 'bout we try out some French cuisine.  Saturday?  Night?"}
    """
    line_dict = {}
    LINE_MAX_PARTS = 5
    for line in lines:
        parts = line.split(" +++$+++ ")
        assert len(parts) == LINE_MAX_PARTS
        line_dict[parts[LinesIdx.LINE_ID]] = parts[LinesIdx.CONVERSATION]
    return line_dict


def _split_conversations_by_marker(conversations: list[str]) -> dict[str, str]:
    """会話コーパスを分割します
    raw:
    u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L194', 'L195', 'L196', 'L197']
    → list(['L194', 'L195', 'L196', 'L197'])
    """
    conversation_list = []
    CONVERSATION_MAX_PARTS = 4
    for conversation in conversations:
        parts = conversation.split(" +++$+++ ")
        assert len(parts) == CONVERSATION_MAX_PARTS
        conversation = parts[ConversationsIdx.CONVERSATION_LIST]
        conversation = conversation.strip("[]\n").replace("'", "").split(", ")
        conversation_list.append(conversation)
    return conversation_list


def generate_q_and_a_pairs():
    """コーパスを読み込んで、質問と回答のペアを生成する。一文が最大64トークンになるようにする。

    Returns:
        list[tuple[str, str]]: 質問と回答のペア
    """
    conversations, lines = _load_datasets()
    line_dict = _split_lines_by_marker(lines)
    conversation_id_list = _split_conversations_by_marker(conversations)
    q_and_a_pairs = []
    for conversation_id in conversation_id_list:
        for i in range(len(conversation_id) - 1):
            # 質問
            first = line_dict[conversation_id[i]].split()[:MAX_LEN]
            # それに対する回答
            second = line_dict[conversation_id[i + 1]].split()[:MAX_LEN]
            q_and_a_pairs.append((" ".join(first), " ".join(second)))
    return q_and_a_pairs


if __name__ == "__main__":
    pairs = generate_q_and_a_pairs()
    for idx, pair in enumerate(pairs[:5]):
        print(f"Q{idx}:", pair[0])
        print(f"A{idx}:", pair[1])
        print()
