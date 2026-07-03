"""
featurize.py
------------
Converts a molecule (given as a SMILES string) into a graph the GNN can
consume: atoms -> nodes, bonds -> edges, with a 3D conformer generated
so bond distances can be used as edge features (same spirit as SchNet /
QM9 in the doc, which use 3D coordinates rather than just topology).

This single function is used in two places:
  1. Building the training dataset (many known molecules -> many graphs)
  2. Predicting a property for a brand-new, arbitrary molecule the user
     supplies at inference time
This guarantees train-time and inference-time features are identical.
"""

import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem

# Elements we explicitly one-hot encode; anything else falls into "other"
ATOM_LIST = ["C", "H", "O", "N", "F", "S", "Cl", "Br"]
BOND_TYPES = [
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
]


def _one_hot(value, choices):
    vec = [0.0] * (len(choices) + 1)  # +1 slot for "other"
    if value in choices:
        vec[choices.index(value)] = 1.0
    else:
        vec[-1] = 1.0
    return vec


def smiles_to_graph(smiles: str) -> Data:
    """
    Returns a torch_geometric.data.Data object with:
      x         : [num_heavy_atoms, node_feat_dim]  atom features
      edge_index: [2, num_bonds*2]                   bonds (both directions)
      edge_attr : [num_bonds*2, edge_feat_dim]        bond type + 3D distance

    NOTE ON DESIGN: graph nodes are HEAVY ATOMS ONLY (standard practice in
    cheminformatics GNNs). Hydrogens are added internally purely so RDKit
    can generate a realistic 3D geometry (bond distances depend on the
    full molecule, hydrogens included) -- but we do not turn every H into
    its own graph node. Instead, each heavy atom gets a "number of
    attached hydrogens" feature. This matters: if hydrogens were their
    own nodes, a molecule's total atom count would be dominated by H's
    (e.g. hexane has 6 heavy atoms but 20 atoms total), which would badly
    confound any pooling operation for properties that don't scale with
    total atom count.

    Raises ValueError if the SMILES string is invalid or a 3D conformer
    cannot be generated.
    """
    mol_heavy = Chem.MolFromSmiles(smiles)
    if mol_heavy is None:
        raise ValueError(f"'{smiles}' is not a valid SMILES string.")

    mol_h = Chem.AddHs(mol_heavy)
    embed_status = AllChem.EmbedMolecule(mol_h, randomSeed=42, useRandomCoords=True)
    if embed_status != 0:
        # retry with a different seed once before giving up
        embed_status = AllChem.EmbedMolecule(mol_h, randomSeed=7, useRandomCoords=True)
    if embed_status != 0:
        raise ValueError(f"Could not generate a 3D conformer for '{smiles}'.")
    AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)

    conf = mol_h.GetConformer()
    positions = conf.GetPositions()  # [num_atoms_incl_H, 3]
    # RDKit's AddHs preserves original heavy-atom indices 0..N-1 and
    # appends new H atoms afterward, so mol_h.GetAtomWithIdx(i) for
    # i < mol_heavy.GetNumAtoms() refers to the same heavy atom.

    n_heavy = mol_heavy.GetNumAtoms()

    # ---- Node features (heavy atoms only) ----
    node_feats = []
    for i in range(n_heavy):
        atom = mol_h.GetAtomWithIdx(i)
        sym = atom.GetSymbol()
        feat = _one_hot(sym, ATOM_LIST)
        feat.append(atom.GetDegree() / 4.0)              # heavy-atom degree
        feat.append(atom.GetTotalNumHs() / 3.0)           # attached H count
        feat.append(atom.GetFormalCharge())
        feat.append(1.0 if atom.GetIsAromatic() else 0.0)
        feat.append(atom.GetAtomicNum() / 35.0)           # rough normalization
        node_feats.append(feat)
    x = torch.tensor(node_feats, dtype=torch.float)

    # ---- Edges (bidirectional, heavy-atom bonds only) + edge features ----
    edge_index = []
    edge_attr = []
    for bond in mol_heavy.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        dist = float(np.linalg.norm(positions[i] - positions[j]))
        bond_feat = _one_hot(bond.GetBondType(), BOND_TYPES)
        bond_feat.append(dist)
        for a, b in [(i, j), (j, i)]:
            edge_index.append([a, b])
            edge_attr.append(bond_feat)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


NODE_FEAT_DIM = len(ATOM_LIST) + 1 + 5   # one-hot atoms+other, degree, numH, charge, aromatic, atomic_num
EDGE_FEAT_DIM = len(BOND_TYPES) + 1 + 1  # one-hot bonds+other, distance
