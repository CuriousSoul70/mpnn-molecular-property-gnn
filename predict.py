"""
predict.py
----------
Loads a trained model and predicts the target property for ANY arbitrary
molecule you give it as a SMILES string. This is the "test any arbitrary
molecule for desired output" piece.

Usage:
    python3 predict.py "CCO"                     # ethanol
    python3 predict.py "c1ccccc1O"                # phenol
    python3 predict.py "CC(=O)OC1=CC=CC=C1C(=O)O" # aspirin
    python3 predict.py "CCO" "c1ccccc1O" "CCN"    # multiple at once
"""

import argparse
import json

import torch

from model import MPNN
from featurize import smiles_to_graph, NODE_FEAT_DIM, EDGE_FEAT_DIM


def load_trained_model(model_path="model.pt", meta_path="model_meta.json"):
    with open(meta_path) as f:
        meta = json.load(f)

    model = MPNN(node_feat_dim=meta["node_feat_dim"],
                 edge_feat_dim=meta["edge_feat_dim"],
                 hidden_dim=64, num_mp_steps=3, readout="sum")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model, meta


def predict_one(model, meta, smiles):
    graph = smiles_to_graph(smiles)
    # give it a fake batch index of all zeros (single molecule = batch of 1)
    graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long)
    with torch.no_grad():
        pred_norm = model(graph)
    pred = pred_norm.item() * meta["target_std"] + meta["target_mean"]
    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("smiles", nargs="+", help="One or more SMILES strings")
    parser.add_argument("--model", default="model.pt")
    parser.add_argument("--meta", default="model_meta.json")
    args = parser.parse_args()

    model, meta = load_trained_model(args.model, args.meta)
    target_name = meta["qm9_target"]

    print(f"Model target property: {target_name}")
    print(f"(trained with test MAE: {meta['test_mae']:.4f})\n")

    for smi in args.smiles:
        try:
            pred = predict_one(model, meta, smi)
            print(f"{smi:40s} -> predicted {target_name}: {pred:.4f}")
        except ValueError as e:
            print(f"{smi:40s} -> ERROR: {e}")


if __name__ == "__main__":
    main()
