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


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch, start_step = 0):
    model.train()
    total_loss = 0.0
    effective_steps = 0
    step_ckpt_interval = 5000

    for step, (x, y) in enumerate(dataloader):
        
        if step < start_step:
            continue
        x, y = x.to(device), y.to(device)
        
        if effective_steps >= max_steps_per_epoch:
            break

        optimizer.zero_grad()

        logits = model(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        effective_steps += 1

        if effective_steps % 1000 == 0:
            print(f"step {effective_steps}, loss = {loss.item():.4f}")
            
        # step-level checkpoint
        if effective_steps > 0 and effective_steps % step_ckpt_interval == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=effective_steps,
                epoch=epoch,
                loss=loss.item(),
                filepath=f"checkpoints/step_ckpt_epoch{epoch + 1}_step{effective_steps}.pt"
            )
            
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=effective_steps,
                epoch=epoch,
                loss=loss.item(),
                filepath="checkpoints/latest.pt"
            )

    return total_loss / max(effective_steps, 1)

# implement scheduler to further reduce loss and perplexity
def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        return max(
            0.01,
            float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
        )
    return LambdaLR(optimizer, lr_lambda)

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
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1)
            )

            total_loss += loss.item()
            steps += 1

            if steps % 500 == 0:
                print(f"eval step {steps}, loss = {loss.item():.4f}")

    return total_loss / max(steps, 1)


def main():
    tokenizer = Tokenizer.from_file("tokenizer/trained_tokenizer/tokenizer.json")

    wikitext = load_dataset("wikitext", "wikitext-2-raw-v1")

    eos_id = tokenizer.token_to_id("<eos>")

    def encode_split(split_name):
        token_ids = []
        for line in wikitext[split_name]["text"]:
            line = line.strip()
            if not line:
                continue

            ids = tokenizer.encode(line).ids
            token_ids.extend(ids)

            if eos_id is not None:
                token_ids.append(eos_id)

        return token_ids

    token_ids_train = encode_split("train")
    token_ids_val = encode_split("validation")
    token_ids_test = encode_split("test")

    print(f"Train tokens: {len(token_ids_train)}")
    print(f"Val tokens: {len(token_ids_val)}")
    print(f"Test tokens: {len(token_ids_test)}")

    train_dataset = LMDataset(token_ids_train, max_seq_len, stride=32)
    val_dataset = LMDataset(token_ids_val, max_seq_len, stride=128)
    test_dataset = LMDataset(token_ids_test, max_seq_len, stride=128)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print("Train dataloader steps per epoch:", len(train_dataloader))
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    run_device = device if torch.cuda.is_available() else "cpu"

    model = CustomerModel(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout
    ).to(run_device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    max_steps_this_epoch = min(len(train_dataloader), max_steps_per_epoch)
    total_training_steps = max_steps_this_epoch * epochs
    scheduler = get_lr_scheduler(optimizer, warmup_steps, total_training_steps)

    start_epoch = 0
    start_step = 0
    checkpoint_path = "checkpoints/latest.pt"
    if os.path.exists(checkpoint_path):
        model, optimizer, start_epoch, start_step, _ = load_checkpoint(
            model,
            optimizer,
            checkpoint_path,
            run_device
        )
        print(f"Resuming from epoch {start_epoch}, step {start_step}")

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    for epoch in range(start_epoch, epochs):
        current_start_step = start_step if epoch == start_epoch else 0

        avg_loss = train_one_epoch(
            model,
            train_dataloader,
            optimizer,
            scheduler,
            run_device,
            epoch,
            start_step=current_start_step
        )

        avg_val_loss = eval_loss(model, val_dataloader, run_device, max_eval_steps=2000)

        train_losses.append(avg_loss)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch + 1}/{epochs}, train_loss = {avg_loss:.4f}, val_loss = {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_gpt_model.pt")
            print(f"Best model saved with val_loss = {best_val_loss:.4f}")

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            step=0,
            epoch=epoch,
            loss=avg_val_loss,
            filepath="checkpoints/latest.pt"
        )

    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("learning_curve.png")
    plt.close()

    model.load_state_dict(torch.load("best_gpt_model.pt", map_location=run_device))
    print("Loaded best model for test evaluation.")
    test_loss = eval_loss(model, test_dataloader, run_device, max_eval_steps=None)
    perplexity = math.exp(test_loss)

    print(f"Test Loss = {test_loss:.4f}")
    print(f"Test Perplexity = {perplexity:.4f}")

    torch.save(model.state_dict(), "gpt_model.pt")
    print("training finished, model saved to gpt_model.pt")

if __name__ == "__main__":
    main()
