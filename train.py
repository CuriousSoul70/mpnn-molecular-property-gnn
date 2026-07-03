"""
train.py
--------
Trains the MPNN on either:
  - the offline demo dataset (default here, since real QM9 download is
    blocked by this sandbox's network whitelist), or
  - real QM9 (set DATASET_MODE = "qm9" -- run this on your own machine
    with normal internet access)

Usage:
    python3 train.py                     # offline demo dataset
    python3 train.py --dataset qm9       # real QM9 (needs internet)
    python3 train.py --dataset qm9 --qm9-target gap --max-molecules 20000
"""

import argparse
import json
import random

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from model import MPNN
from featurize import NODE_FEAT_DIM, EDGE_FEAT_DIM

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def split_dataset(data_list, train_frac=0.8, val_frac=0.1):
    data_list = list(data_list)
    random.shuffle(data_list)
    n = len(data_list)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return (data_list[:n_train],
            data_list[n_train:n_train + n_val],
            data_list[n_train + n_val:])


def compute_normalization(train_list):
    ys = torch.stack([d.y.view(-1) for d in train_list]).view(-1)
    return ys.mean().item(), ys.std().item()


def mae_eval(model, loader, mean, std, device):
    model.eval()
    total_abs_err = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred_norm = model(batch)
            pred = pred_norm * std + mean
            target = batch.y.view(-1) * std + mean
            total_abs_err += (pred - target).abs().sum().item()
            n += target.numel()
    return total_abs_err / n


def train(dataset_mode="offline", qm9_target="gap", max_molecules=None,
          epochs=60, batch_size=32, lr=1e-3, target_mae=None,
          out_prefix="model"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if dataset_mode == "offline":
        import os
        cache_path = "cached_dataset.pt"
        if os.path.exists(cache_path):
            print(f"Loading cached offline dataset from {cache_path} ...")
            data_list, smiles_list, _ = torch.load(cache_path, weights_only=False)
        else:
            from dataset_synthetic import build_offline_dataset
            print("Building OFFLINE demo dataset (real molecules, RDKit-computed "
                  "TPSA as the target) -- real QM9 download is blocked on this "
                  "sandbox's network. Use --dataset qm9 on your own machine "
                  "for real quantum-chemistry targets.")
            data_list, smiles_list, _ = build_offline_dataset(n_target=600)
            torch.save((data_list, smiles_list, _), cache_path)
        print(f"Using {len(data_list)} valid molecules.")
    elif dataset_mode == "qm9":
        from qm9_loader import load_qm9, QM9_TARGET_NAMES
        target_index = QM9_TARGET_NAMES.index(qm9_target)
        data_list = load_qm9(target_index=target_index, max_molecules=max_molecules)
    else:
        raise ValueError(f"Unknown dataset_mode: {dataset_mode}")

    train_list, val_list, test_list = split_dataset(data_list)
    print(f"Train/Val/Test sizes: {len(train_list)}/{len(val_list)}/{len(test_list)}")

    mean, std = compute_normalization(train_list)
    print(f"Target normalization: mean={mean:.4f}, std={std:.4f}")

    for d in train_list + val_list + test_list:
        d.y = (d.y.view(-1) - mean) / std

    train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=batch_size)
    test_loader = DataLoader(test_list, batch_size=batch_size)

    model = MPNN(node_feat_dim=NODE_FEAT_DIM, edge_feat_dim=EDGE_FEAT_DIM,
                 hidden_dim=64, num_mp_steps=3, readout="sum").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = torch.nn.MSELoss()

    best_val_mae = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = loss_fn(pred, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        train_loss = total_loss / len(train_list)
        val_mae = mae_eval(model, val_loader, mean, std, device)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train MSE (norm): {train_loss:.4f} "
                  f"| val MAE (real units): {val_mae:.4f}")

        if target_mae is not None and val_mae <= target_mae:
            print(f"Reached target MAE ({target_mae}) at epoch {epoch}. Stopping early.")
            break

    model.load_state_dict(best_state)
    test_mae = mae_eval(model, test_loader, mean, std, device)
    print(f"\nFinal test MAE (real units): {test_mae:.4f}")
    print(f"Best val MAE (real units):   {best_val_mae:.4f}")

    torch.save(model.state_dict(), f"{out_prefix}.pt")
    with open(f"{out_prefix}_meta.json", "w") as f:
        json.dump({
            "dataset_mode": dataset_mode,
            "qm9_target": qm9_target if dataset_mode == "qm9" else "TPSA (offline demo)",
            "target_mean": mean,
            "target_std": std,
            "node_feat_dim": NODE_FEAT_DIM,
            "edge_feat_dim": EDGE_FEAT_DIM,
            "test_mae": test_mae,
        }, f, indent=2)

    print(f"Saved model weights to {out_prefix}.pt and metadata to {out_prefix}_meta.json")
    return model, mean, std, test_mae


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["offline", "qm9"], default="offline")
    parser.add_argument("--qm9-target", default="gap",
                         help="Which QM9 property to predict (only used with --dataset qm9)")
    parser.add_argument("--max-molecules", type=int, default=None,
                         help="Cap on number of QM9 molecules (only used with --dataset qm9)")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--target-mae", type=float, default=None,
                         help="Stop training early once validation MAE reaches this value")
    parser.add_argument("--out-prefix", default="model")
    args = parser.parse_args()

    train(dataset_mode=args.dataset, qm9_target=args.qm9_target,
          max_molecules=args.max_molecules, epochs=args.epochs,
          batch_size=args.batch_size, lr=args.lr,
          target_mae=args.target_mae, out_prefix=args.out_prefix)
