import os
import glob
import json
import argparse
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from xyzrender import load, render

from model import EGNNVectorField, SizePredictor
from fm import OCFM

ATOM_MAP_INV = {0: 'H', 1: 'C', 2: 'N', 3: 'O', 4: 'F'}

def generate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default=None, help="Directory containing model and config")
    parser.add_argument("--gap", type=float, default=5.0, help="Target HOMO-LUMO gap in eV")
    parser.add_argument("--num_mols", type=int, default=5, help="Number of molecules to generate")
    parser.add_argument("--out_dir", type=str, default="generated", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="Adjacency threshold for bonding")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.run_dir is None:
        runs = sorted(glob.glob("results/run_*"))
        if not runs: raise ValueError("No run directories found.")
        args.run_dir = runs[-1]
        
    with open(os.path.join(args.run_dir, "config.json"), "r") as f:
        config = json.load(f)
        
    # 1. Models
    model = EGNNVectorField(in_node_dim=5, hidden_dim=config['hidden_dim'], num_layers=config['num_layers']).to(device)
    size_predictor = SizePredictor(cond_dim=config['hidden_dim'], max_nodes=config['max_nodes']).to(device)
    
    ckpt = torch.load(os.path.join(args.run_dir, "nano_flow_mol.pt"), map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model']); size_predictor.load_state_dict(ckpt['size_predictor'])
    model.eval(); size_predictor.eval()
    
    # 2. Normalize and Sample sizes
    norm_gap = (args.gap - config["y_mean"]) / config["y_std"]
    y_cond = torch.full((args.num_mols, 1), norm_gap, device=device)
    
    with torch.no_grad():
        mu, log_var = size_predictor(y_cond)
        # Sample from predicted Gaussian and round
        std = torch.exp(0.5 * log_var)
        num_nodes_list = torch.normal(mu, std).round().long().clamp(1, config['max_nodes']).squeeze().tolist()
        if args.num_mols == 1: num_nodes_list = [num_nodes_list]

    # 3. Generate
    fm = OCFM(prior_scale=config.get("prior_scale", 0.7))
    print(f"Generating {args.num_mols} molecules (sizes: {num_nodes_list})...")
    
    gen_x, gen_h, gen_e, gen_batch, gen_edges = fm.generate(model, num_nodes_list, y_cond, device, steps=100)
    
    atom_types = gen_h.argmax(dim=-1)
    os.makedirs(args.out_dir, exist_ok=True)
    
    valid_count = 0
    for i in range(args.num_mols):
        mask = (gen_batch == i)
        x_i = gen_x[mask].cpu().numpy()
        types_i = atom_types[mask].cpu().numpy()
        
        # Predicted bonds for this molecule
        e_mask = (gen_batch[gen_edges[0]] == i)
        e_i = gen_e[e_mask]
        edges_i = gen_edges[:, e_mask]
        # Map global batch indices to local 0..N-1 indices
        local_map = {idx.item(): j for j, idx in enumerate(torch.where(mask)[0])}
        
        # Build RWMol with connectivity
        mol = Chem.RWMol()
        for t in types_i:
            mol.AddAtom(Chem.Atom(ATOM_MAP_INV.get(t.item(), 'C')))
            
        # Add bonds based on predicted adjacency threshold
        for val, u, v in zip(e_i, edges_i[0], edges_i[1]):
            if val > args.threshold and u < v: # Avoid double edges
                try:
                    mol.AddBond(local_map[u.item()], local_map[v.item()], Chem.BondType.SINGLE)
                except: pass
                
        # Finalize and set coords
        try:
            res_mol = mol.GetMol()
            conf = Chem.Conformer(len(x_i))
            for j, pos in enumerate(x_i):
                conf.SetAtomPosition(j, (float(pos[0]), float(pos[1]), float(pos[2])))
            res_mol.AddConformer(conf)
            
            # Use RDKit to refine bonds (it will use our connectivity as a skeleton)
            rdDetermineBonds.DetermineBondOrders(res_mol)
            valid_count += 1
            status = "VALID"
        except:
            status = "INVALID"
            res_mol = None

        # Save XYZ
        xyz_path = os.path.join(args.out_dir, f"mol_{i}.xyz")
        with open(xyz_path, "w") as f:
            f.write(f"{len(x_i)}\nGap {args.gap} eV\n")
            for j in range(len(x_i)):
                f.write(f"{ATOM_MAP_INV.get(types_i[j].item(), 'C')} {x_i[j][0]:.4f} {x_i[j][1]:.4f} {x_i[j][2]:.4f}\n")
        
        print(f"Molecule {i} (Size {len(x_i)}): {status}")
        try:
            render(load(xyz_path), output=os.path.join(args.out_dir, f"mol_{i}.png"))
        except: pass
            
    print(f"\nGeneration complete! Validity: {valid_count}/{args.num_mols}")

if __name__ == "__main__":
    generate()
