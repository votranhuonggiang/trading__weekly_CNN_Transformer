from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.model import CNNTransformerClassifier


@dataclass
class TrainingConfig:
    epochs: int = 12
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 3
    checkpoint_dir: str = "checkpoints"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_class_weights: bool = True


def train_model(
    arrays: dict[str, np.ndarray],
    config: TrainingConfig | None = None,
) -> tuple[CNNTransformerClassifier, dict[str, float | str | dict]]:
    config = config or TrainingConfig()
    device = torch.device(config.device)

    train_loader = _make_loader(arrays["X_train"], arrays["y_train"], config.batch_size, shuffle=True)
    val_loader = _make_loader(arrays["X_val"], arrays["y_val"], config.batch_size, shuffle=False)
    test_loader = _make_loader(arrays["X_test"], arrays["y_test"], config.batch_size, shuffle=False)

    num_classes = int(np.max(arrays["y_train"])) + 1
    model = CNNTransformerClassifier(num_features=arrays["X_train"].shape[-1], num_classes=num_classes).to(device)
    class_weights = _balanced_class_weights(arrays["y_train"], num_classes).to(device) if config.use_class_weights else None
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_cnn_transformer.pt"

    best_val_f1 = -1.0
    best_epoch = -1
    wait = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        train_loss = _run_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate_model(model, val_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            wait = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            wait += 1
            if wait >= config.patience:
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_metrics = evaluate_model(model, test_loader, device)
    test_metrics["best_epoch"] = best_epoch
    test_metrics["checkpoint_path"] = str(checkpoint_path)
    test_metrics["history"] = history
    return model, test_metrics


def evaluate_model(
    model: CNNTransformerClassifier,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float | str]:
    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            logits = model(x_batch.to(device))
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            preds.append(pred)
            targets.append(y_batch.numpy())

    y_true = np.concatenate(targets)
    y_pred = np.concatenate(preds)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "y_true": y_true,
        "y_pred": y_pred,
        "report": classification_report(
            y_true,
            y_pred,
            target_names=["Avoid", "Hold", "Buy"],
            digits=4,
            zero_division=0,
        ),
    }


def _make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _balanced_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def _run_epoch(
    model: CNNTransformerClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    losses: list[float] = []
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    return float(np.mean(losses))
