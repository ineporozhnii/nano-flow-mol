"""Ground-truth check of gap conditioning with DFT at the QM9 level of theory (B3LYP/6-31G(2df,p)).

Pipeline per generated molecule (one JSON per molecule, so runs can be interrupted and resumed):
  generated geometry --(MMFF94 "conformation")--> --(B3LYP/6-31G(2df,p) geometry optimization)--> gap
recording whether the covalent bond graph changes at each relaxation step, whether it has an O-O bond,
and (optionally, if a regressor checkpoint exists) a proxy-regressor gap for the raw generated geometry.

Modes:
  --calibrate N    single points on N QM9 test molecules; compares with the QM9 gap column
  (default)        generate molecules at --targets and run the pipeline on them
Use --shard i/k to split the work across several processes (e.g. one per GPU).
"""
import os
import sys
import json
import time
import argparse

if "--cpu" not in sys.argv:
    import cupy
    cupy.linalg.eigh(cupy.eye(2))

import numpy as np
import torch

from chem import SYMBOLS, check_stability, mol_to_smiles, build_mol, connectivity, has_oo_bond
from data import HARTREE_TO_EV

XC, BASIS = "b3lypg", "6-31g(2df,p)"
DENSITY_FIT = True


def make_mf(symbols, pos, gpu=True, grid_level=4):
    from pyscf import gto
    mol = gto.M(atom=[(s, tuple(map(float, p))) for s, p in zip(symbols, pos)],
                basis=BASIS, charge=0, spin=0, unit="Angstrom", verbose=0)
    if gpu:
        from gpu4pyscf.dft import rks
    else:
        from pyscf.dft import rks
    mf = rks.RKS(mol, xc=XC)
    mf.grids.level = grid_level
    mf.conv_tol = 1e-9
    mf.max_cycle = 200
    return mf.density_fit() if DENSITY_FIT else mf


def gap_from_mf(mf):
    mo_e = np.asarray(mf.mo_energy.get() if hasattr(mf.mo_energy, "get") else mf.mo_energy)
    mo_occ = np.asarray(mf.mo_occ.get() if hasattr(mf.mo_occ, "get") else mf.mo_occ)
    homo = mo_e[mo_occ > 0].max()
    lumo = mo_e[mo_occ == 0].min()
    return float((lumo - homo) * HARTREE_TO_EV), float(homo * HARTREE_TO_EV), float(lumo * HARTREE_TO_EV)


def single_point(symbols, pos, gpu=True):
    mf = make_mf(symbols, pos, gpu)
    energy = float(mf.kernel())
    gap, homo, lumo = gap_from_mf(mf)
    return {"energy_ha": energy, "gap_ev": gap, "homo_ev": homo, "lumo_ev": lumo, "scf_converged": bool(mf.converged)}


def optimize(symbols, pos, gpu=True, max_steps=100):
    from pyscf.geomopt.geometric_solver import optimize as geometric_opt
    mf = make_mf(symbols, pos, gpu, grid_level=3)  # final gap is recomputed on the level-4 grid
    # Gaussian-default convergence criteria; QM9 geometries were optimized with Gaussian
    mol_eq = geometric_opt(mf, maxsteps=max_steps, convergence_set="GAU")
    return np.asarray(mol_eq.atom_coords(unit="Angstrom"))


def mmff_relax(symbols, pos):
    """MMFF94 relaxation of the generated geometry (bonds perceived from distances).
    Returns relaxed positions, or None if MMFF cannot be set up."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = build_mol(symbols, pos)
    try:
        Chem.SanitizeMol(mol)
        if not AllChem.MMFFHasAllMoleculeParams(mol):
            return None
        AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    except Exception:
        return None
    return mol.GetConformer().GetPositions()


def evaluate_molecule(rec, gpu=True):
    symbols, pos0 = rec["symbols"], np.array(rec["pos"])
    t0 = time.time()
    out = dict(rec)
    out["oo_bond"] = bool(has_oo_bond(symbols, pos0))
    g0 = connectivity(symbols, pos0)

    pos1 = mmff_relax(symbols, pos0)
    out["mmff_ok"] = pos1 is not None
    if pos1 is None:
        pos1 = pos0
    g1 = connectivity(symbols, pos1)
    out["graph_changed_conformation"] = g1 != g0

    out["sp_raw"] = single_point(symbols, pos0, gpu)  # gap on the raw generated geometry
    pos2 = optimize(symbols, pos1, gpu)
    g2 = connectivity(symbols, pos2)
    out["graph_changed_relaxation"] = g2 != g1
    out["dft_opt"] = single_point(symbols, pos2, gpu)
    out["pos_opt"] = pos2.tolist()
    out["smiles_opt"] = mol_to_smiles(symbols, pos2)
    out["time_s"] = time.time() - t0
    return out


def calibrate(n, gpu, out_path):
    from data import get_datasets
    _, _, test_ds, _, _ = get_datasets(normalize=False)
    rows = []
    for i in range(n):
        d = test_ds[i]
        symbols = [SYMBOLS[t] for t in d.x.argmax(-1).tolist()]
        t0 = time.time()
        sp = single_point(symbols, d.pos.numpy(), gpu)
        rows.append({"smiles": d.smiles, "n_atoms": len(symbols), "qm9_gap_ev": float(d.y),
                     "pyscf_gap_ev": sp["gap_ev"], "abs_err_ev": abs(sp["gap_ev"] - float(d.y)),
                     "time_s": time.time() - t0})
        print(json.dumps(rows[-1]), flush=True)
    t0 = time.time()
    optimize(symbols, d.pos.numpy() + np.random.default_rng(0).normal(0, 0.05, d.pos.shape), gpu)
    summary = {"mean_abs_err_ev": float(np.mean([r["abs_err_ev"] for r in rows])),
               "max_abs_err_ev": float(np.max([r["abs_err_ev"] for r in rows])),
               "opt_time_s_perturbed_last": time.time() - t0, "rows": rows}
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)


def build_candidates(args, path):
    """Generate stable, unique molecules for each target and store them (fixed set for all shards)."""
    from generate import load_run, sample_many
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    model, size_pred, fm, config = load_run(args.run_dir, device)
    use_proxy = args.regressor and os.path.exists(args.regressor)
    if use_proxy:  # optional: proxy-predicted gaps for comparison with DFT
        from regressor import load_regressor, predict_gaps
        reg, r_mean, r_std = load_regressor(args.regressor, device)
    records = []
    for t in args.targets:
        mols = sample_many(model, size_pred, fm, config, [t] * args.n_generate, device)
        seen = set()
        kept = []
        for symbols, pos in mols:
            smi = mol_to_smiles(symbols, pos)
            if smi is None or smi in seen or not check_stability(symbols, pos)[0]:
                continue
            seen.add(smi)
            kept.append((symbols, pos, smi))
            if len(kept) == args.n_per_target:
                break
        proxy = (predict_gaps(reg, r_mean, r_std, [(s, p) for s, p, _ in kept], device).tolist()
                 if use_proxy else [None] * len(kept))
        print(f"target {t} eV: kept {len(kept)} stable+unique of {len(mols)} generated")
        for k, ((symbols, pos, smi), pg) in enumerate(zip(kept, proxy)):
            records.append({"id": f"t{t:g}_{k:03d}", "target_ev": t, "symbols": symbols,
                            "pos": np.asarray(pos).tolist(), "smiles": smi, "proxy_gap_ev": pg})
    meta = {"checkpoint": config["checkpoint"], "regressor": args.regressor if use_proxy else None,
            "targets": args.targets, "seed": args.seed, "n_generate": args.n_generate, "xc": XC, "basis": BASIS, "density_fit": DENSITY_FIT, "records": records}
    with open(path, "w") as f:
        json.dump(meta, f)
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", default=None)
    p.add_argument("--regressor", default="checkpoints/regressor.pt",
                   help="optional proxy regressor; skipped if the file does not exist")
    p.add_argument("--targets", type=float, nargs="+", default=[4.2, 6.8, 9.4])
    p.add_argument("--n_per_target", type=int, default=64)
    p.add_argument("--n_generate", type=int, default=3000, help="samples drawn per target before filtering")
    p.add_argument("--out_dir", default="results/dft")
    p.add_argument("--shard", default="0/1", help="i/k: process every k-th molecule starting at i")
    p.add_argument("--cpu", action="store_true", help="use CPU PySCF instead of GPU4PySCF")
    p.add_argument("--no_df", action="store_true", help="exact integrals instead of density fitting (slow)")
    p.add_argument("--calibrate", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="process at most this many molecules (testing)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    gpu = not args.cpu
    global DENSITY_FIT
    DENSITY_FIT = not args.no_df
    os.makedirs(args.out_dir, exist_ok=True)

    if args.calibrate:
        calibrate(args.calibrate, gpu, os.path.join(args.out_dir, "calibration.json"))
        return

    cand_path = os.path.join(args.out_dir, "candidates.json")
    if os.path.exists(cand_path):
        with open(cand_path) as f:
            meta = json.load(f)
    elif args.shard == "0/1" or args.shard.startswith("0/"):
        meta = build_candidates(args, cand_path)
    else:
        sys.exit("candidates.json missing: run shard 0 first (it generates the molecule set).")

    i, k = map(int, args.shard.split("/"))
    todo = meta["records"][i::k][:args.limit]
    mol_dir = os.path.join(args.out_dir, "molecules")
    os.makedirs(mol_dir, exist_ok=True)
    for n, rec in enumerate(todo):
        path = os.path.join(mol_dir, f"{rec['id']}.json")
        if os.path.exists(path):
            continue
        try:
            out = evaluate_molecule(rec, gpu)
        except Exception as e:  # SCF/optimizer failure: record it and move on
            out = {**rec, "error": f"{type(e).__name__}: {e}"}
        with open(path + ".tmp", "w") as f:
            json.dump(out, f)
        os.replace(path + ".tmp", path)
        gap = out.get("dft_opt", {}).get("gap_ev")
        proxy = rec.get("proxy_gap_ev")
        print(f"[{n + 1}/{len(todo)}] {rec['id']} target {rec['target_ev']} | "
              f"proxy {'-' if proxy is None else round(proxy, 2)} | "
              f"DFT {gap if gap is None else round(gap, 2)} | {out.get('time_s', 0):.0f}s", flush=True)


if __name__ == "__main__":
    main()
