import os
import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from checkpoint import save_checkpoint, load_checkpoint
from tokenizers import Tokenizer
from datasets import load_dataset

from config import *
from transformer.CustomerModel import CustomerModel
from dataset import LMDataset


def train_one_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    effective_steps = 0

    for _, (x, y) in enumerate(dataloader):
        if effective_steps >= max_steps_per_epoch:
            break

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        logits = model(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        effective_steps += 1

        if effective_steps % 1000 == 0:
            print(f"step {effective_steps}, loss = {loss.item():.4f}")

    return total_loss / max(effective_steps, 1)


def get_lr_scheduler(optimizer, warmup_steps, total_steps, last_epoch=-1, eta_min_ratio=0.01):
    if total_steps < 1:
        raise ValueError(f"total_steps must be >= 1, got {total_steps}.")
    warmup_steps = max(0, warmup_steps)

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step + 1) / float(max(warmup_steps, 1))
        if step >= total_steps:
            return float(eta_min_ratio)
        i = step - warmup_steps
        cosine_steps = total_steps - warmup_steps
        if cosine_steps <= 0:
            return 1.0
        progress = float(i) / float(max(1, cosine_steps - 1))
        progress = min(1.0, progress)
        return float(eta_min_ratio) + (1.0 - float(eta_min_ratio)) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)

def eval_loss(model, dataloader, device, max_eval_steps=None):
    model.eval()
    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        for step, (x, y) in enumerate(dataloader):
            if max_eval_steps is not None and step >= max_eval_steps:
                break

            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
            )
            total_loss += loss.item()
            steps += 1

            if steps % 500 == 0:
                print(f"eval step {steps}, loss = {loss.item():.4f}")

    return total_loss / max(steps, 1)


def _encode_wikitext_split(wikitext, split_name, tokenizer, eos_id):
    token_ids = []
    for line in wikitext[split_name]["text"]:
        line = line.strip()
        if not line:
            continue
        token_ids.extend(tokenizer.encode(line).ids)
        if eos_id is not None:
            token_ids.append(eos_id)
    return token_ids


def main():
    tokenizer = Tokenizer.from_file("tokenizer/trained_tokenizer/tokenizer1.json")
    tok_vocab = tokenizer.get_vocab_size()
    if tok_vocab != vocab_size:
        raise ValueError(
            f"config vocab_size ({vocab_size}) != tokenizer vocab size ({tok_vocab})."
        )

    wikitext = load_dataset("wikitext", "wikitext-103-raw-v1")
    eos_id = tokenizer.token_to_id("<eos>")

    token_ids_train = _encode_wikitext_split(wikitext, "train", tokenizer, eos_id)
    token_ids_val = _encode_wikitext_split(wikitext, "validation", tokenizer, eos_id)
    token_ids_test = _encode_wikitext_split(wikitext, "test", tokenizer, eos_id)

    print(f"Train tokens: {len(token_ids_train)}")
    print(f"Val tokens: {len(token_ids_val)}")
    print(f"Test tokens: {len(token_ids_test)}")

    train_dataset = LMDataset(token_ids_train, max_seq_len, stride=max_seq_len)
    val_dataset = LMDataset(token_ids_val, max_seq_len, stride=max_seq_len)
    test_dataset = LMDataset(token_ids_test, max_seq_len, stride=max_seq_len)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    max_steps_this_epoch = min(len(train_dataloader), max_steps_per_epoch)
    if max_steps_this_epoch < 1:
        raise RuntimeError(
            "No training steps: empty train set or batch_size too large for data length."
        )

    run_device = device if torch.cuda.is_available() else "cpu"

    model = CustomerModel(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
    ).to(run_device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_training_steps = max_steps_this_epoch * epochs
    if total_training_steps < 1:
        raise RuntimeError("epochs must be >= 1 when there is at least one step per epoch.")

    start_epoch = 0
    checkpoint_path = "checkpoints/latest.pt"
    if os.path.exists(checkpoint_path):
        model, optimizer, start_epoch, _ = load_checkpoint(
            model,
            optimizer,
            checkpoint_path,
            run_device,
        )
        print(f"Resuming at epoch {start_epoch + 1}/{epochs}")

    completed_opt_steps = start_epoch * max_steps_this_epoch
    scheduler_last_epoch = completed_opt_steps - 1 if completed_opt_steps > 0 else -1
    scheduler = get_lr_scheduler(
        optimizer,
        warmup_steps,
        total_training_steps,
        last_epoch=scheduler_last_epoch,
    )

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_path = "best_gpt_model.pt"

    print("Train dataloader steps per epoch (capped):", max_steps_this_epoch)

    for epoch in range(start_epoch, epochs):
        avg_loss = train_one_epoch(
            model,
            train_dataloader,
            optimizer,
            scheduler,
            run_device,
        )

        avg_val_loss = eval_loss(model, val_dataloader, run_device, max_eval_steps=2000)

        train_losses.append(avg_loss)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch + 1}/{epochs}, train_loss = {avg_loss:.4f}, val_loss = {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_path)
            print(f"Best model saved with val_loss = {best_val_loss:.4f}")

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            next_epoch=epoch + 1,
            loss=avg_val_loss,
            filepath=checkpoint_path,
        )

    if train_losses:
        plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss")
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Learning Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig("learning_curve.png")
        plt.close()

    if not os.path.isfile(best_path):
        raise FileNotFoundError(
            f"{best_path} not found; training may not have completed any epoch."
        )

    model.load_state_dict(torch.load(best_path, map_location=run_device))
    print("Loaded best model for test evaluation.")
    test_loss = eval_loss(model, test_dataloader, run_device, max_eval_steps=None)
    try:
        perplexity = math.exp(test_loss)
    except OverflowError:
        perplexity = float("inf")

    print(f"Test Loss = {test_loss:.4f}")
    print(f"Test Perplexity = {perplexity:.4f}")

    torch.save(model.state_dict(), "gpt_model.pt")
    print("training finished, model saved to gpt_model.pt")

if __name__ == "__main__":
    main()
