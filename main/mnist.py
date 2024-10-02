from __future__ import annotations

from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from typing import Iterator, Sized

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.jit.script
def criterion(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.nll_loss(torch.log_softmax(logits, dim=1), labels)


def train(model: nn.Module, optim: Optimizer, scheduler: LRScheduler, loader: Iterator[tuple[torch.Tensor, torch.Tensor]]) -> tuple[float, float]:
    model.train()
    x, l = next(loader)
    optim.zero_grad(set_to_none=True)
    logits = model(x)
    loss = criterion(logits, l)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1)
    optim.step()
    scheduler.step()
    acc = (torch.argmax(logits, dim=1) == l).sum() / x.size(0)
    return loss.item(), acc.item()


@torch.inference_mode()
def valid(model: nn.Module, loader: DataLoader, dataset: Sized) -> tuple[float, float]:
    model.eval()
    loss = torch.tensor(0, dtype=torch.float32)
    acc = torch.tensor(0, dtype=torch.float32)
    for x, l in loader:
        logits = model(x)
        loss += criterion(logits, l)
        acc += (torch.argmax(logits, dim=1) == l).sum().item()
    return loss.item() / len(loader), acc.item() / len(dataset)


if __name__ == "__main__":
    from itertools import cycle
    from torch.optim.adamw import AdamW
    from torch.optim.lr_scheduler import OneCycleLR
    from torch.utils.data import Subset
    from torchvision.datasets.mnist import MNIST
    from tqdm import tqdm

    import matplotlib.pyplot as plt
    import torch.onnx as tonnx
    import torchvision.transforms as T


    # ========== Hyperparameters
    epochs = 20
    valid_every_n_step = 100
    train_valid_frac = 0.8
    batch_size = 1_024
    lr = 1e-3

    # ========== Dataset Splits
    transform = T.ToTensor()

    dataset = MNIST("/tmp/datasets", train=True, download=True, transform=transform)
    valid_indices = range(int(train_valid_frac * len(dataset)), len(dataset))
    train_indices = range(0, int(train_valid_frac * len(dataset)))

    valid_set = Subset(dataset, indices=valid_indices)
    train_set = Subset(dataset, indices=train_indices)
    test_set = MNIST("/tmp/datasets", train=False, download=True, transform=transform)

    # ========== Dataloaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


    # ========== Setup Model, Optimizer, and LRScheduler
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(1 * 28 * 28, 512), nn.ReLU(inplace=True),
        nn.Linear(512, 512), nn.ReLU(inplace=True),
        nn.Linear(512, 10),
    )
    optim = AdamW(model.parameters(), lr, fused=True)
    scheduler = OneCycleLR(optim, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs)

    # ========== Training Loop w/ Validation
    valid_loss, valid_acc = 0, 0
    train_history, valid_history, lr_history = [], [], []
    train_batches = iter(cycle(train_loader))
    pbar = tqdm(range(epochs * len(train_loader)), desc="Train")
    for step in pbar:
        train_loss, train_acc = train(model, optim, scheduler, train_batches)
        train_history.append(train_loss)
        lr_history.append(scheduler.get_last_lr()[0])

        if step % valid_every_n_step == 0:
            valid_loss, valid_acc = valid(model, valid_loader, valid_set)
            valid_history.append(valid_loss)

        pbar.set_postfix(
            lr=f"{scheduler.get_last_lr()[0]:.2e}",
            train_loss=f"{train_loss:.2e}",
            train_acc=f"{train_acc * 100:.2f}%",
            valid_loss=f"{valid_loss:.2e}",
            valid_acc=f"{valid_acc * 100:.2f}%",
        )

    # ========== Monitoring Plot
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(0, epochs * len(train_loader)), train_history, label="train")
    plt.plot(range(0, epochs * len(train_loader), valid_every_n_step), valid_history, label="valid")
    plt.legend()
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.subplot(1, 2, 2)
    plt.plot(range(0, epochs * len(train_loader)), lr_history, label="lr")
    plt.legend()
    plt.xlabel("steps")
    plt.ylabel("lr")
    plt.tight_layout()
    plt.savefig("mnist.png")

    # ========== Test
    test_loss, test_acc = valid(model, test_loader, test_set)
    print(f"Test Loss: {test_loss:.2e}")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    # ========== Save the Model Weights
    torch.save(model.state_dict(), "mnist.pt")
    tonnx.export(model, torch.empty((1, 1, 28, 28)), "mnist.onnx", input_names=["input"], output_names=["output"])
