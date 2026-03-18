import torch
import os

# Save training checkpoint.
# Args:
#         model: GPT model
#         step: current step before termination
#         optimizer: optimizer (e.g. AdamW)
#         epoch: current epoch number
#         loss: current loss
#         filepath: path to save checkpoint
def save_checkpoint(model, optimizer, step, epoch, loss, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }

    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")

#  Load training checkpoint.
#  Args:
        # model: GPT model
        # optimizer: optimizer
        # filepath: checkpoint path
        # device: cpu or cuda

def load_checkpoint(model, optimizer, filepath, device):
    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_step = checkpoint["step"]
    start_epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]

    print(f"Checkpoint loaded from {filepath}")

    return model, optimizer, start_epoch, start_step, loss
