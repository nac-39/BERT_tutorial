import torch
from bert_tutorial import (
    BERT,
    BERTLM,
    BERTTrainer,
    BERTDataset,
    generate_q_and_a_pairs,
    init_tokenizer,
)
from torch.utils.data import DataLoader, random_split


MAX_LEN = 64

if __name__ == "__main__":
    train = True
    pairs = generate_q_and_a_pairs()
    tokenizer = init_tokenizer(pairs)
    dataset = BERTDataset(pairs, seq_len=MAX_LEN, tokenizer=tokenizer, device="cuda")
    n_samples = len(dataset)
    train_size = int(n_samples * 0.8)
    val_size = n_samples - train_size
    train_data, test_data = random_split(dataset, [train_size, val_size])

    if train:
        train_loader = DataLoader(
            train_data, batch_size=32, shuffle=True, pin_memory=True
        )
        bert_model = BERT(
            vocab_size=len(tokenizer.vocab),
            d_model=768,
            n_layers=2,
            heads=12,
            dropout=0.1,
        )

        bert_lm = BERTLM(bert_model, len(tokenizer.vocab))
        bert_trainer = BERTTrainer(bert_lm, train_loader, device="cuda")
        epochs = 1

    for epoch in range(epochs):
        bert_trainer.train(epoch)
    else:
        test_loader = DataLoader(
            test_data, batch_size=32, shuffle=False, pin_memory=True
        )
        bert_model = BERT(
            vocab_size=len(tokenizer.vocab),
            d_model=768,
            n_layers=2,
            heads=12,
            dropout=0.1,
        )

        bert_lm = BERTLM(bert_model, len(tokenizer.vocab))
        bert_trainer = BERTTrainer(bert_lm,train_dataloader=train_loader, test_dataloader=test_loader, device="cuda")
        bert_trainer.model.load_state_dict(torch.load("bert_pretrain_epoch_0.pt"))
        epochs = 1

        with torch.no_grad():
            for epoch in range(epochs):
                bert_trainer.test(epoch)
