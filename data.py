"""QM9 loading: raw .xyz files -> PyG InMemoryDataset with positions, one-hot atom types,
HOMO-LUMO gap (eV) and canonical SMILES, plus a seeded train/val/test split."""
import os
import glob

import torch
from tqdm import tqdm
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

HARTREE_TO_EV = 27.211386
ATOM_MAP = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
ATOM_MAP_INV = {v: k for k, v in ATOM_MAP.items()}
NUM_ATOM_TYPES = len(ATOM_MAP)
GAP_COLUMN = 9  # index of the gap (Hartree) in the QM9 property line, after the "gdb <id>" tag


def _float(s):
    # QM9 uses Mathematica-style exponents, e.g. "1.2*^-6"
    return float(s.replace('*^', 'e'))


def parse_qm9_xyz(file_path):
    """Returns (symbols, positions [N,3], gap_ev, smiles) for one QM9 .xyz file."""
    with open(file_path) as f:
        lines = f.readlines()
    n = int(lines[0])
    props = lines[1].split()
    gap_ev = _float(props[GAP_COLUMN]) * HARTREE_TO_EV
    symbols, pos = [], []
    for line in lines[2:2 + n]:
        parts = line.split()
        symbols.append(parts[0])
        pos.append([_float(p) for p in parts[1:4]])
    # line n+3 holds "GDB-SMILES  B3LYP-relaxed-SMILES"; use the relaxed one
    smiles = lines[3 + n].split()[-1]
    return symbols, pos, gap_ev, smiles


class QM9FlowDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['dsgdb9nsd_000001.xyz']

    @property
    def processed_file_names(self):
        return ['qm9_flow_v2.pt']

    def download(self):
        raise FileNotFoundError(
            f"QM9 raw files not found in {self.raw_dir}. Run `python download_dataset.py` first.")

    def process(self):
        file_paths = sorted(glob.glob(os.path.join(self.raw_dir, "*.xyz")))
        if not file_paths:
            raise FileNotFoundError(f"No .xyz files found in {self.raw_dir}.")

        data_list = []
        for file_path in tqdm(file_paths, desc="Processing QM9"):
            symbols, pos, gap_ev, smiles = parse_qm9_xyz(file_path)
            pos = torch.tensor(pos, dtype=torch.float32)
            pos = pos - pos.mean(dim=0, keepdim=True)
            types = torch.tensor([ATOM_MAP[s] for s in symbols])
            h = torch.nn.functional.one_hot(types, NUM_ATOM_TYPES).float()
            data_list.append(Data(pos=pos, x=h, y=torch.tensor([gap_ev]), smiles=smiles))

        print(f"Saving {len(data_list)} molecules.")
        self.save(data_list, self.processed_paths[0])


def get_split(n, root_dir='./data/qm9', seed=42, val_frac=0.05, test_frac=0.10):
    """Deterministic train/val/test indices, cached to disk so every script sees the same split."""
    path = os.path.join(root_dir, f'split_seed{seed}.pt')
    if os.path.exists(path):
        split = torch.load(path, weights_only=True)
        if sum(len(v) for v in split.values()) == n:
            return split
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_val, n_test = int(n * val_frac), int(n * test_frac)
    split = {'test': perm[:n_test], 'val': perm[n_test:n_test + n_val], 'train': perm[n_test + n_val:]}
    torch.save(split, path)
    return split


def get_datasets(root_dir='./data/qm9', seed=42, normalize=True):
    """Returns (train, val, test) datasets and the train-set gap mean/std (eV).
    If normalize=True, `y` is standardized with the train statistics."""
    dataset = QM9FlowDataset(root=root_dir)
    split = get_split(len(dataset), root_dir, seed)
    y = dataset._data.y
    y_mean, y_std = y[split['train']].mean().item(), y[split['train']].std().item()
    if normalize:
        dataset._data.y = (y - y_mean) / y_std
    return dataset[split['train']], dataset[split['val']], dataset[split['test']], y_mean, y_std


def get_dataloaders(root_dir='./data/qm9', batch_size=64, num_workers=4, seed=42):
    train_ds, val_ds, _, y_mean, y_std = get_datasets(root_dir, seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, y_mean, y_std


if __name__ == "__main__":
    train_ds, val_ds, test_ds, m, s = get_datasets()
    print(f"train {len(train_ds)} | val {len(val_ds)} | test {len(test_ds)} | gap {m:.3f} ± {s:.3f} eV")
