import os

import torch
import torch.nn.functional as F

def compute_metrics(logits, labels):
    loss = F.cross_entropy(logits, labels, reduction="mean")
    accuracy = (logits.argmax(dim=1) == labels).float().mean()

    metrics = {
        "loss": loss,
        "accuracy": accuracy
    }
    return metrics

def train_step(model, batch, optimizer, device):
    model.train()

    images = batch["image"].to(device)
    labels = batch["label"].to(device)

    optimizer.zero_grad()

    logits = model(images)
    loss = F.cross_entropy(logits, labels)

    loss.backward()
    optimizer.step()

    metrics = compute_metrics(logits, labels)
    return metrics

def train_loop(model, train_loader, optimizer, epoch, device):
    batch_metrics = []

    for batch in train_loader:
        metrics = train_step(model, batch, optimizer, device)
        batch_metrics.append(metrics)

    epoch_metrics = {
        k: torch.stack([m[k] for m in batch_metrics]).mean().item()
        for k in batch_metrics[0]
    }

    print(
        f"EPOCH: {epoch}\n"
        f"Training loss: {epoch_metrics['loss']:.4f}, "
        f"accuracy: {epoch_metrics['accuracy'] * 100:.2f}"
    )
    return epoch_metrics
    
@torch.no_grad()
def eval_loop(model, test_loader, device):
    batch_metrics = []

    for batch in test_loader:
        metrics = eval_step(model, batch, device)
        batch_metrics.append(metrics)

    epoch_metrics = {
        k: torch.stack([m[k] for m in batch_metrics]).mean().item()
        for k in batch_metrics[0]
    }

    print(
        f"    Eval loss: {epoch_metrics['loss']:.4f}, "
        f"accuracy: {epoch_metrics['accuracy'] * 100:.2f}"
    )
    return epoch_metrics

    
@torch.no_grad()
def eval_step(model, batch, device):
    model.eval()

    images = batch["image"].to(device)
    labels = batch["label"].to(device)

    logits = model(images)
    metrics = compute_metrics(logits, labels)
    return metrics

@torch.no_grad()
def pred_step(model, batch, device):
    model.eval()
    batch = batch.to(device)
    logits = model(batch)
    return logits.argmax(dim=1)


def save_checkpoint(model, optimizer, epoch, path, max_to_keep=3):
    ckpt_file = os.path.join(path, f"epoch_{epoch}.pt")

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, ckpt_file)
    
    ckpts = sorted(
        [f for f in os.listdir(path) if f.endswith(".pt")],
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )

    if len(ckpts) > max_to_keep:
        os.remove(os.path.join(path, ckpts[0]))

def save_checkpoint(model, optimizer, epoch, path, max_to_keep=3):
    ckpt_file = os.path.join(path, f"epoch_{epoch}.pt")

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, ckpt_file)
    
    ckpts = sorted(
        [f for f in os.listdir(path) if f.endswith(".pt")],
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )

    if len(ckpts) > max_to_keep:
        os.remove(os.path.join(path, ckpts[0]))

def save_checkpoint(model, optimizer, epoch, path, max_to_keep=3):
    ckpt_file = os.path.join(path, f"epoch_{epoch}.pt")

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, ckpt_file)
    
    ckpts = sorted(
        [f for f in os.listdir(path) if f.endswith(".pt")],
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )

    if len(ckpts) > max_to_keep:
        os.remove(os.path.join(path, ckpts[0]))

def load_checkpoint(path,device):
    ckpts = sorted(
        [f for f in os.listdir(path) if f.endswith(".pt")],
        key=lambda x: int(x.split("_")[1].split(".")[0]),
        reverse=True
    )
    
    ckpt_file = os.path.join(path, ckpts[0])

    checkpoint = torch.load(ckpt_file, map_location=device)
    
    return checkpoint
    
