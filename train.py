import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
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


def eval_loss(model, dataloader, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )
            total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    pattern = [1,2,3,4,5,6,7,8]
    token_ids = pattern * 500
    dataset = LMDataset(token_ids, max_seq_len)


    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # MODIFIED
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # ADDED
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # ADDED

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

    train_losses = []  # ADDED
    val_losses = []  # ADDED

    for epoch in range(epochs):
        avg_loss = train_one_epoch(model, dataloader, optimizer, run_device)

        avg_val_loss = eval_loss(model, val_dataloader, run_device)

        train_losses.append(avg_loss) 
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch + 1}/{epochs}, train_loss = {avg_loss:.4f}, val_loss = {avg_val_loss:.4f}")

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss=avg_loss,
            filepath=f"checkpoints/gpt_epoch_{epoch + 1}.pt"
        )


    plt.plot(range(1, epochs + 1), train_losses, label="Training Loss")
    plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("learning_curve.png")
    plt.show()

    test_loss = eval_loss(model, test_dataloader, run_device)
    perplexity = math.exp(test_loss)

    print(f"Test Loss = {test_loss:.4f}")
    print(f"Test Perplexity = {perplexity:.4f}")

    torch.save(model.state_dict(), "gpt_model.pt")
    print("training finished, model saved to gpt_model.pt")


if __name__ == "__main__":
    main()
