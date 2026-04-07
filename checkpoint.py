import os

import torch


def save_checkpoint(model, optimizer, next_epoch, loss, filepath):
    """Save training state. Resume with load_checkpoint; training continues from next_epoch."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    checkpoint = {
        "next_epoch": next_epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }

    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, optimizer, filepath, device):
    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if "next_epoch" in checkpoint:
        start_epoch = checkpoint["next_epoch"]
    else:
        # Legacy: "epoch" was the loop index when saving at epoch end (e.g. 0 after finishing epoch 0).
        start_epoch = checkpoint["epoch"] + 1

    loss = checkpoint["loss"]
    print(f"Checkpoint loaded from {filepath}")

    return model, optimizer, start_epoch, loss
