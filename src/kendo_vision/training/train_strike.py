"""Trains StrikeNet on labelled recordings."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from ..config import Config
from ..events.strikenet import StrikeNet
from .constants import HEADS, UNKNOWN
from .strike_dataset import StrikeDataset


def _loss(outputs, target, attacker, ippon):
    """Target loss everywhere; attacker and ippon losses only where they are labelled."""
    target_logits, attacker_logits, ippon_logits = outputs
    loss = functional.cross_entropy(target_logits, target)
    if (attacker != UNKNOWN).any():  # cross_entropy returns NaN when every label is ignored
        loss = loss + functional.cross_entropy(attacker_logits, attacker, ignore_index=UNKNOWN)
    ippon_known = ippon != UNKNOWN
    if ippon_known.any():
        loss = loss + functional.binary_cross_entropy_with_logits(ippon_logits[ippon_known], ippon[ippon_known])
    return loss


@torch.no_grad()
def _evaluate(network, loader, device) -> dict[str, float]:
    """Mean loss, and accuracy of each head over the samples that have a label for it."""
    network.eval()
    total_loss, sample_count = 0.0, 0
    correct = {head: 0 for head in HEADS}
    labelled = {head: 0 for head in HEADS}
    for windows, target, attacker, ippon in loader:
        windows, target = windows.to(device), target.to(device)
        attacker, ippon = attacker.to(device), ippon.to(device)
        outputs = network(windows)
        target_logits, attacker_logits, ippon_logits = outputs
        total_loss += float(_loss(outputs, target, attacker, ippon)) * len(windows)
        sample_count += len(windows)
        has_label = {"target": torch.ones_like(target, dtype=torch.bool),
                     "attacker": attacker != UNKNOWN, "ippon": ippon != UNKNOWN}
        is_right = {"target": target_logits.argmax(dim=-1) == target,
                    "attacker": attacker_logits.argmax(dim=-1) == attacker,
                    "ippon": (ippon_logits > 0) == (ippon > 0.5)}  # a logit above 0 means "ippon"
        for head in HEADS:
            correct[head] += int((is_right[head] & has_label[head]).sum())
            labelled[head] += int(has_label[head].sum())
    metrics = {"loss": total_loss / max(sample_count, 1)}
    for head in HEADS:
        metrics[head] = correct[head] / labelled[head] if labelled[head] else float("nan")
    return metrics


def train(data: str, validation_names: list[str], config: Config, epochs: int = 60,
          learning_rate: float = 1e-3, batch_size: int = 32, output: str | None = None) -> None:
    """`validation_names` are the recordings held out, so the split is by match rather than by window."""
    folder = Path(data)
    # a recording counts only if it has both its features and its labels
    names = sorted(path.stem for path in folder.glob("*.npz") if (folder / f"{path.stem}.jsonl").exists())
    training_names = [name for name in names if name not in validation_names]
    held_out_names = [name for name in names if name in validation_names]
    if not training_names:
        raise SystemExit(f"No <name>.npz + <name>.jsonl pairs to train on in {folder}")
    if not held_out_names:
        print("No validation recordings named; reporting on the training set, which flatters the model.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    training_loader = DataLoader(StrikeDataset(training_names, folder, config, train=True),
                                 batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(StrikeDataset(held_out_names or training_names, folder, config, train=False),
                                   batch_size=batch_size)
    network = StrikeNet().to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    output = output or config.model.strikenet_weights
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    for epoch in range(epochs):
        network.train()
        for windows, target, attacker, ippon in training_loader:
            optimizer.zero_grad()
            outputs = network(windows.to(device))
            _loss(outputs, target.to(device), attacker.to(device), ippon.to(device)).backward()
            optimizer.step()
        scheduler.step()
        metrics = _evaluate(network, validation_loader, device)
        improved = metrics["loss"] < best_loss
        if improved:  # keep the weights from the best epoch, not the last
            best_loss = metrics["loss"]
            torch.save(network.state_dict(), output)
        print(f"epoch {epoch + 1:3}/{epochs}  val loss {metrics['loss']:.3f}  target {metrics['target']:.2f}  "
              f"attacker {metrics['attacker']:.2f}  ippon {metrics['ippon']:.2f}" + ("  saved" if improved else ""))
    print(f"Best weights in {output}. Set validity.mode: learned to use them.")
