"""
qm9_loader.py
-------------
Loads the REAL QM9 dataset (134k small organic molecules, each with 19
properties originally computed via DFT quantum chemistry calculations).

This is what you should use on your own machine. It needs normal internet
access because torch_geometric downloads a ~2GB file from data.pyg.org
the first time you run it (cached locally after that).

QM9 target properties (column index into data.y), the most commonly
predicted ones:
    0  mu       - dipole moment
    1  alpha    - isotropic polarizability
    2  homo     - energy of HOMO
    3  lumo     - energy of LUMO
    4  gap      - HOMO-LUMO gap        <-- good default target
    5  r2       - electronic spatial extent
    6  zpve     - zero point vibrational energy
    7  u0       - internal energy at 0K
    ... (see PyG docs for the full list of 19)
"""

from torch_geometric.datasets import QM9
from featurize import smiles_to_graph

QM9_TARGET_NAMES = [
    "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "u0", "u298",
    "h298", "g298", "cv", "u0_atom", "u298_atom", "h298_atom",
    "g298_atom", "A", "B", "C",
]


def load_qm9(root="./data/qm9", target_index=4, max_molecules=None):
    """
    Downloads (first run only) and loads QM9.

    IMPORTANT DESIGN CHOICE: rather than using PyG's built-in QM9 graph
    encoding directly, we re-featurize each molecule's SMILES string with
    our own featurize.smiles_to_graph(). This costs a bit of extra
    preprocessing time, but guarantees that a model trained on real QM9
    uses IDENTICAL input features to a model trained on the offline demo
    dataset -- so predict.py works unmodified either way, and you can
    swap between them freely. The regression TARGET (y) is still the
    real DFT-computed QM9 value; only the 3D geometry used as model
    *input* is re-derived via RDKit/MMFF instead of QM9's stored DFT
    geometry.

    target_index: which of the 19 properties to predict (default 4 = gap,
                  the HOMO-LUMO gap -- matches the doc's emphasis on
                  reaction/energetics-relevant quantities).
    max_molecules: optional cap, useful for a quick first run since QM9
                   has 134k molecules and re-featurizing all of them
                   takes a while.
    """
    raw_dataset = QM9(root=root)
    print(f"Downloaded/loaded real QM9: {len(raw_dataset)} molecules.")
    print(f"Predicting target: {QM9_TARGET_NAMES[target_index]} "
          f"(column {target_index})")

    n = len(raw_dataset) if max_molecules is None else min(max_molecules, len(raw_dataset))

    data_list = []
    skipped = 0
    for i in range(n):
        raw = raw_dataset[i]
        smi = raw.smiles
        try:
            graph = smiles_to_graph(smi)
        except ValueError:
            skipped += 1
            continue
        graph.y = raw.y[:, target_index].clone()
        graph.smiles = smi
        data_list.append(graph)

    print(f"Re-featurized {len(data_list)} molecules "
          f"({skipped} skipped due to 3D embedding failures).")
    return data_list
