"""Molecule utilities: bond perception, stability/validity checks and RDKit conversion.

Stability and validity follow the EDM protocol (Hoogeboom et al., 2022): bonds are assigned from
interatomic distances using reference bond lengths, an atom is stable if its total bond order matches
its valence, and a molecule is stable if all atoms are stable. Validity = RDKit sanitization succeeds.
"""
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDetermineBonds

RDLogger.DisableLog('rdApp.*')

SYMBOLS = ['H', 'C', 'N', 'O', 'F']

# reference bond lengths in pm (EDM, bond_analyze.py)
BONDS1 = {'H': {'H': 74, 'C': 109, 'N': 101, 'O': 96, 'F': 92},
          'C': {'H': 109, 'C': 154, 'N': 147, 'O': 143, 'F': 135},
          'N': {'H': 101, 'C': 147, 'N': 145, 'O': 140, 'F': 136},
          'O': {'H': 96, 'C': 143, 'N': 140, 'O': 148, 'F': 142},
          'F': {'H': 92, 'C': 135, 'N': 136, 'O': 142, 'F': 142}}
BONDS2 = {'C': {'C': 134, 'N': 129, 'O': 120}, 'N': {'C': 129, 'N': 125, 'O': 121},
          'O': {'C': 120, 'N': 121, 'O': 121}}
BONDS3 = {'C': {'C': 120, 'N': 116, 'O': 113}, 'N': {'C': 116, 'N': 110}, 'O': {'C': 113}}
MARGINS = (10, 5, 3)
VALENCE = {'H': 1, 'C': 4, 'N': 3, 'O': 2, 'F': 1}
RDKIT_BOND = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}


def bond_order(a1, a2, dist_angstrom):
    d = 100.0 * dist_angstrom
    if d >= BONDS1[a1][a2] + MARGINS[0]:
        return 0
    if a2 in BONDS2.get(a1, {}) and d < BONDS2[a1][a2] + MARGINS[1]:
        if a2 in BONDS3.get(a1, {}) and d < BONDS3[a1][a2] + MARGINS[2]:
            return 3
        return 2
    return 1


def bond_matrix(symbols, pos):
    pos = np.asarray(pos, dtype=float)
    n = len(symbols)
    dist = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    bonds = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            bonds[i, j] = bonds[j, i] = bond_order(symbols[i], symbols[j], dist[i, j])
    return bonds


def check_stability(symbols, pos):
    """Returns (molecule_stable, n_stable_atoms, n_atoms)."""
    valence = bond_matrix(symbols, pos).sum(1)
    n_stable = sum(int(v == VALENCE[s]) for s, v in zip(symbols, valence))
    return n_stable == len(symbols), n_stable, len(symbols)


def build_mol(symbols, pos):
    """RDKit molecule with EDM-style distance bonds and a 3D conformer (unsanitized)."""
    bonds = bond_matrix(symbols, pos)
    mol = Chem.RWMol()
    for s in symbols:
        mol.AddAtom(Chem.Atom(s))
    for i, j in zip(*np.triu_indices(len(symbols), 1)):
        if bonds[i, j]:
            mol.AddBond(int(i), int(j), RDKIT_BOND[int(bonds[i, j])])
    conf = Chem.Conformer(len(symbols))
    for i, p in enumerate(pos):
        conf.SetAtomPosition(i, [float(c) for c in p])
    mol.AddConformer(conf, assignId=True)
    return mol.GetMol()


def mol_to_smiles(symbols, pos, largest_fragment=True):
    """Canonical SMILES if the molecule sanitizes (largest fragment by default), else None."""
    mol = build_mol(symbols, pos)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    if largest_fragment:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        mol = max(frags, key=lambda m: m.GetNumAtoms())
    return canonical(mol)


def canonical(mol_or_smiles):
    mol = Chem.MolFromSmiles(mol_or_smiles) if isinstance(mol_or_smiles, str) else mol_or_smiles
    if mol is None:
        return None
    return Chem.MolToSmiles(Chem.RemoveHs(mol, sanitize=False), isomericSmiles=False)


def connectivity(symbols, pos):
    """Covalent-radius connectivity graph (frozenset of atom-index pairs); None on failure.
    Used to detect topology changes during force-field / DFT relaxation."""
    mol = Chem.RWMol()
    for s in symbols:
        mol.AddAtom(Chem.Atom(s))
    conf = Chem.Conformer(len(symbols))
    for i, p in enumerate(pos):
        conf.SetAtomPosition(i, [float(c) for c in p])
    mol.AddConformer(conf)
    try:
        rdDetermineBonds.DetermineConnectivity(mol)
    except Exception:
        return None
    return frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds())


def has_oo_bond(symbols, pos):
    bonds = bond_matrix(symbols, pos)
    return any(bonds[i, j] and symbols[i] == symbols[j] == 'O'
               for i in range(len(symbols)) for j in range(i + 1, len(symbols)))


def write_xyz(path, symbols, pos, comment=""):
    with open(path, "w") as f:
        f.write(f"{len(symbols)}\n{comment}\n")
        for s, p in zip(symbols, pos):
            f.write(f"{s} {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")


def split_batch(x, h, batch):
    """Split batched generator output into a list of (symbols, positions[N,3] numpy)."""
    types = h.argmax(-1).cpu().numpy()
    x = x.cpu().numpy()
    batch = batch.cpu().numpy()
    out = []
    for i in range(int(batch.max()) + 1):
        m = batch == i
        out.append(([SYMBOLS[t] for t in types[m]], x[m]))
    return out
