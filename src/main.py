from bert_tutorial import (
    BERT,
    BERTLM,
    BERTTrainer,
    BERTDataset,
    generate_q_and_a_pairs,
    init_tokenizer,
)
from torch.utils.data import DataLoader


MAX_LEN = 64

if __name__ == "__main__":
    pairs = generate_q_and_a_pairs()
    tokenizer = init_tokenizer(pairs)
    train_data = BERTDataset(pairs, seq_len=MAX_LEN, tokenizer=tokenizer)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, pin_memory=True)

    bert_model = BERT(
        vocab_size=len(tokenizer.vocab), d_model=768, n_layers=2, heads=12, dropout=0.1
    )

    bert_lm = BERTLM(bert_model, len(tokenizer.vocab))
    bert_trainer = BERTTrainer(bert_lm, train_loader, device="cuda")
    epochs = 20

    for epoch in range(epochs):
        bert_trainer.train(epoch)
