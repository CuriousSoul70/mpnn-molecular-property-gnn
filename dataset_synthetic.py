"""
dataset_synthetic.py
---------------------
WHY THIS FILE EXISTS:
Real QM9 is downloaded by torch_geometric from data.pyg.org. This sandbox's
network is restricted to a whitelist of domains (pypi, github, etc.) that
does not include data.pyg.org, so the download fails here with a 403.
On your own machine (normal internet access), qm9_loader.py below will
just work and you should use that instead.

To still let you see and run the FULL pipeline right now, this file builds
an offline dataset of several hundred real, valid small organic molecules
(combinatorially generated, then filtered with RDKit for chemical
validity) and computes a REAL topological/physicochemical property for
each one using RDKit -- specifically Topological Polar Surface Area (TPSA),
a standard cheminformatics descriptor (not fabricated numbers, not a
quantum property, just a fast classical descriptor that correlates with
molecular structure -- good enough to prove the GNN can actually learn
structure -> property mappings end to end).

Swap DATASET_MODE to "qm9" in train.py once you have real internet
access, and the exact same model/training code will run on the real
quantum-chemistry-derived QM9 targets instead.
"""

import random
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from torch_geometric.data import InMemoryDataset

from featurize import smiles_to_graph

random.seed(42)

# Building blocks for combinatorial generation of small, valid organic
# molecules covering a range of functional groups (alkanes, alcohols,
# amines, ethers, halides, carbonyls, aromatics, acids...).
CHAINS = ["C", "CC", "CCC", "CCCC", "CCCCC", "CCCCCC",
          "CC(C)C", "CC(C)CC", "CC(C)(C)C"]
FUNCTIONAL_GROUPS = [
    "", "O", "N", "F", "Cl", "Br", "C=O", "C(=O)O", "C#N",
    "OC", "N(C)C", "C(=O)N",
]
RINGS = [
    "c1ccccc1", "c1ccncc1", "C1CCCCC1", "c1ccc(O)cc1",
    "c1ccc(N)cc1", "C1CCOC1", "c1ccc(Cl)cc1", "c1ccc(C)cc1",
]


def _generate_candidate_smiles(n_target=400):
    candidates = set()

    # Chain + functional group combinations
    for chain in CHAINS:
        for fg in FUNCTIONAL_GROUPS:
            candidates.add(chain + fg)
            candidates.add(fg + chain if fg else chain)

    # Ring compounds, some plain, some with an attached chain
    for ring in RINGS:
        candidates.add(ring)
        for chain in CHAINS[:5]:
            candidates.add(ring.replace("1)", "1)") + chain)  # simple append attempt

    # Two functional groups on one chain (much more chemical diversity)
    for chain in CHAINS:
        for fg1 in FUNCTIONAL_GROUPS:
            for fg2 in FUNCTIONAL_GROUPS:
                if fg1 or fg2:
                    candidates.add(fg1 + chain + fg2)

    # Ring + two substituents
    for ring in RINGS:
        for fg1 in FUNCTIONAL_GROUPS:
            for chain in CHAINS[:6]:
                candidates.add(ring + fg1 + chain)
                candidates.add(fg1 + ring + chain)

    # Random multi-part combinations for extra diversity
    combos = list(candidates)
    for _ in range(n_target * 6):
        a = random.choice(combos)
        b = random.choice(FUNCTIONAL_GROUPS)
        c = random.choice(FUNCTIONAL_GROUPS)
        candidates.add(a + b + c)

    return list(candidates)


def _valid_unique_molecules(candidates, n_target):
    seen_inchi = set()
    kept = []
    for smi in candidates:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        if mol.GetNumAtoms() < 2 or mol.GetNumAtoms() > 24:
            continue
        try:
            inchi = Chem.MolToInchiKey(mol)
        except Exception:
            continue
        if inchi in seen_inchi:
            continue
        seen_inchi.add(inchi)
        kept.append(Chem.MolToSmiles(mol))
        if len(kept) >= n_target:
            break
    return kept


def build_offline_dataset(n_target=350, target_property="TPSA"):
    """
    Returns (data_list, smiles_list, raw_targets) where data_list is a
    list of torch_geometric Data graphs ready for training, and
    raw_targets are the real RDKit-computed property values (unnormalized).
    """
    candidates = _generate_candidate_smiles(n_target)
    smiles_list = _valid_unique_molecules(candidates, n_target)

    prop_fn = {
        "TPSA": Descriptors.TPSA,
        "MolLogP": Descriptors.MolLogP,
        "MolWt": Descriptors.MolWt,
    }[target_property]

    data_list = []
    kept_smiles = []
    raw_targets = []
    for smi in smiles_list:
        try:
            graph = smiles_to_graph(smi)
        except ValueError:
            continue
        mol = Chem.MolFromSmiles(smi)
        y = float(prop_fn(mol))
        graph.y = torch.tensor([y], dtype=torch.float)
        graph.smiles = smi
        data_list.append(graph)
        kept_smiles.append(smi)
        raw_targets.append(y)

    return data_list, kept_smiles, raw_targets


class OfflineMoleculeDataset(InMemoryDataset):
    """Thin InMemoryDataset wrapper so it behaves like any other PyG dataset."""

    def __init__(self, data_list):
        super().__init__(None)
        self.data, self.slices = self.collate(data_list)

    def _download(self):
        pass

    def _process(self):
        pass
