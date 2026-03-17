import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from checkpoint import save_checkpoint

from config import *
from transformer.PreTrainingModel import PreTrainingModel
from dataset import LMDataset

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    for step, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        logits = model(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if step % 50 == 0:
            print(f"step {step}, loss = {loss.item():.4f}")

    return total_loss / len(dataloader)

def main():
    # following tokenizer and dataset should be modified
    pattern = [1,2,3,4,5,6,7,8]
    token_ids = pattern * 500
    dataset = LMDataset(token_ids, max_seq_len) # need to change
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    run_device = device if torch.cuda.is_available() else "cpu"

    model = PreTrainingModel(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout
    ).to(run_device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        avg_loss = train_one_epoch(model, dataloader, optimizer, run_device)
        print(f"Epoch {epoch + 1}/{epochs}, avg_loss = {avg_loss:.4f}")

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss=avg_loss,
            filepath=f"checkpoints/gpt_epoch_{epoch + 1}.pt"
        )

    torch.save(model.state_dict(), "gpt_model.pt")
    print("training finished, model saved to gpt_model.pt")

if __name__ == "__main__":
    main()