import os
import glob
import torch
from tqdm import tqdm
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

# Standard QM9 Atom Types
ATOM_MAP = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
NUM_ATOM_TYPES = len(ATOM_MAP)

class QM9FlowDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return ['dsgdb9nsd_000001.xyz']

    @property
    def processed_file_names(self):
        return ['qm9_flow_sparse_bonds.pt']

    def download(self):
        pass

    def process(self):
        file_paths = glob.glob(os.path.join(self.raw_dir, "*.xyz"))
        if len(file_paths) == 0:
            raise FileNotFoundError(f"No .xyz files found in {self.raw_dir}.")

        data_list = []
        skipped = 0
        
        print(f"Processing {len(file_paths)} molecules with robust bond extraction...")
        for file_path in tqdm(file_paths):
            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    
                num_atoms = int(lines[0].strip())
                props = lines[1].strip().split()
                # props[9] is gap in Hartree
                gap_ev = float(props[9]) * 27.2114
                
                pos_list = []
                h_list = []
                
                # We build the RDKit molecule manually instead of using MolFromXYZBlock
                # to avoid parser issues with the QM9 properties line.
                mol = Chem.RWMol()
                conf = Chem.Conformer(num_atoms)
                
                for j in range(num_atoms):
                    parts = lines[2 + j].strip().split()
                    symbol = parts[0]
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    
                    # Add to RDKit mol
                    idx = mol.AddAtom(Chem.Atom(symbol))
                    conf.SetAtomPosition(idx, (x, y, z))
                    
                    # Add to lists
                    pos_list.append([x, y, z])
                    one_hot = [0.0] * NUM_ATOM_TYPES
                    one_hot[ATOM_MAP[symbol]] = 1.0
                    h_list.append(one_hot)

                res_mol = mol.GetMol()
                res_mol.AddConformer(conf)
                
                pos = torch.tensor(pos_list, dtype=torch.float32)
                h = torch.tensor(h_list, dtype=torch.float32)
                
                # Center coordinates
                pos = pos - pos.mean(dim=0, keepdim=True)
                
                data = Data(pos=pos, x=h, y=torch.tensor([gap_ev]))
                data_list.append(data)
                
            except Exception as e:
                skipped += 1
                continue

        print(f"Saving {len(data_list)} molecules. Skipped: {skipped}")
        if len(data_list) == 0:
            raise RuntimeError("No molecules processed successfully.")
            
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def get_dataloaders(root_dir='./data/qm9', batch_size=64, val_split=0.1, num_workers=4):
    dataset = QM9FlowDataset(root=root_dir)
    
    # Use recommended access to attributes to suppress warnings
    y_all = dataset.y
    y_mean, y_std = y_all.mean(), y_all.std()
    # Apply normalization to the internal tensor
    dataset._data.y = (y_all - y_mean) / y_std
    
    dataset = dataset.shuffle()
    val_size = int(len(dataset) * val_split)
    train_dataset = dataset[val_size:]
    val_dataset = dataset[:val_size]
    
    # Adding num_workers to saturate GPU
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, y_mean, y_std

if __name__ == "__main__":
    dataset = QM9FlowDataset(root="./data/qm9")
    print(f"Success. Dataset size: {len(dataset)}")
