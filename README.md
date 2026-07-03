# MPNN Molecular Property GNN

A from-scratch **Message Passing Neural Network (MPNN)** for predicting
molecular properties from structure, built and benchmarked on QM9-style
data. This is a classical AI/DL baseline — no quantum computing involved
anywhere in this code — built as part of a larger research project
exploring quantum-enhanced methods in drug discovery.

## Is QM9 a "quantum" dataset?

Not in the sense of quantum *computing*. QM9 is a dataset of ~134,000
small organic molecules where each one has properties (dipole moment,
HOMO-LUMO gap, atomization energy, etc.) that were originally *computed*
using DFT — a quantum *chemistry* method, run on a classical computer.
The dataset itself is just molecular structures + numbers, and it's the
standard benchmark for classical ML models (SchNet, MPNN, and similar
graph neural networks were all built and tested on it).

## What this repo does

1. **Loads molecular data** — either a real QM9 subset (needs internet;
   see `qm9_loader.py`) or a built-in offline demo dataset of ~600 real,
   RDKit-validated small organic molecules (`dataset_synthetic.py`)
2. **Converts any molecule into a graph** the model can consume — atoms
   as nodes, bonds as edges, 3D interatomic distances as edge features
   (`featurize.py`)
3. **Trains an MPNN** to predict a scalar property from that graph
   (`model.py`, `train.py`)
4. **Predicts on any arbitrary molecule** you give it as a SMILES string
   — including molecules never seen during training (`predict.py`)

## Quickstart

```bash
git clone https://github.com/<your-username>/mpnn-molecular-property-gnn.git
cd mpnn-molecular-property-gnn
pip install -r requirements.txt

# Train on the offline demo dataset (works immediately, no internet needed
# beyond the initial pip install)
python3 train.py --epochs 100 --target-mae 4.0

# Predict the property for any molecule
python3 predict.py "CCO" "c1ccccc1O" "CC(=O)OC1=CC=CC=C1C(=O)O"
```

## Repo structure

| File | Purpose |
|---|---|
| `featurize.py` | Converts a SMILES string into a graph (nodes, edges, features). Used identically at train time and inference time. |
| `model.py` | The MPNN architecture — atoms exchange messages shaped by bond/distance features, then a readout produces a molecule-level prediction. |
| `dataset_synthetic.py` | Builds the offline demo dataset (real molecules, RDKit-computed Topological Polar Surface Area as the target). |
| `qm9_loader.py` | Downloads and loads **real QM9** (needs internet) — same featurization pipeline as the offline dataset, so it's a drop-in swap. |
| `train.py` | Training loop: train/val/test split, MAE tracking, early stopping, checkpoint saving. |
| `predict.py` | Loads a trained model and predicts the target property for any arbitrary molecule. |
| `model.pt` / `model_meta.json` | A trained checkpoint (offline demo dataset) included so you can run `predict.py` immediately without retraining. |

## Results (offline demo dataset)

600 molecules, 480/60/60 train/val/test split, predicting TPSA
(Topological Polar Surface Area):

- **Test MAE: 5.15** vs. a "always predict the mean" baseline of ~24 —
  the model is genuinely learning structure→property relationships, not
  memorizing the average.

On molecules never seen during training:

| Molecule | Predicted TPSA | True TPSA | Error |
|---|---|---|---|
| ethanol | 34.97 | 20.23 | 14.74 |
| phenol | 23.60 | 20.23 | 3.37 |
| aspirin | 75.91 | 63.60 | 12.31 |
| triethylamine | 36.68 | 3.24 | 33.44 |

**Honest limitation:** the offline demo dataset (~600 combinatorially
generated molecules) doesn't cover the full diversity of real-world
chemistry — it under-represents tertiary amines, which is why
triethylamine's prediction is so far off. This is expected and disclosed
rather than hidden. Real QM9's 134,000 molecules give dramatically
better coverage.

## Using real QM9

```bash
python3 train.py --dataset qm9 --qm9-target gap --max-molecules 20000
python3 predict.py "CCO" "c1ccccc1O"
```

- `--qm9-target` selects which of QM9's 19 DFT-computed properties to
  predict (`gap` = HOMO-LUMO gap, the default; others include `mu`,
  `alpha`, `homo`, `lumo`, `zpve`, `u0`, `cv`, ...)
- `--max-molecules` caps how many of QM9's 134k molecules to use —
  useful for a fast first run
- First run downloads ~2GB via `torch_geometric` (cached afterward)
- Predictions will then be in real physical units (e.g. eV for the
  HOMO-LUMO gap) — genuine DFT-grounded quantum chemistry properties,
  learned by a purely classical model

## Design notes

- **Heavy-atom-only graphs**: hydrogens are added internally only to get
  a realistic 3D geometry (via RDKit/MMFF), but are not their own graph
  nodes — each heavy atom carries a "number of attached hydrogens"
  feature instead. Treating H atoms as separate nodes badly confounds
  sum-pooling for any property that doesn't scale with total atom count.
- **Sum-pooling, not mean-pooling**: TPSA (like most QM9 targets, e.g.
  energies) is an *extensive* property — it scales with molecule size.
  Mean-pooling erases that signal.
- **MPNN tier specifically**: this is deliberately the "message passing
  neural network" architecture (edge-conditioned convolution + GRU
  update), not the more complex SchNet/DimeNet/GemNet tier. It's a
  correct, well-understood baseline — a natural next step is upgrading
  to continuous-filter convolutions (SchNet-style) once this is solid.

## Requirements

```
torch
torch_geometric
rdkit
numpy
```

## License

MIT — see `LICENSE`.
