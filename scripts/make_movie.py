"""Render a GIF of molecules emerging from noise along the learned flow, one panel per target gap.

With --from_dft (default when results/dft exists), each panel shows a DFT-verified molecule: the
generation run of dft_eval.py is replayed with the same seed to recover its trajectories, and for each
target the molecule whose relaxed DFT gap is closest to the target (unchanged bond graph) is shown."""
import os
import sys
import json
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image

import plot_style
from plot_style import plt
from chem import SYMBOLS, check_stability, mol_to_smiles
from generate import load_run, sample

CPK = {"H": "#f4f4f2", "C": "#3c3c3c", "N": "#3050f8", "O": "#e8221b", "F": "#6cc24a"}
RADIUS = {"H": 0.25, "C": 0.40, "N": 0.38, "O": 0.37, "F": 0.35}      # drawn radius (Å)
COVALENT = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57}


def rotation_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def principal_axes(pos):
    """Rotation that puts the largest spread along x and the smallest along z (viewing axis)."""
    _, _, vt = np.linalg.svd(pos - pos.mean(0))
    r = vt.T
    if np.linalg.det(r) < 0:
        r[:, 2] *= -1
    return r


def draw_molecule(ax, symbols, pos, bond_alpha, extent):
    ax.clear()
    ax.set_xlim(-extent, extent); ax.set_ylim(-extent, extent)
    ax.set_aspect("equal"); ax.axis("off")
    n = len(symbols)
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    if bond_alpha > 0:
        for i in range(n):
            for j in range(i + 1, n):
                if d[i, j] < 1.15 * (COVALENT[symbols[i]] + COVALENT[symbols[j]]):
                    ax.plot(pos[[i, j], 0], pos[[i, j], 1], color="#8a8984", lw=3.2,
                            alpha=bond_alpha, solid_capstyle="round", zorder=1)
    zmin, zmax = pos[:, 2].min(), pos[:, 2].max() + 1e-6
    for i in np.argsort(pos[:, 2]):  # back to front
        depth = (pos[i, 2] - zmin) / (zmax - zmin)        # 0 = far, 1 = near
        ax.add_patch(plt.Circle(pos[i, :2], RADIUS[symbols[i]] * (0.85 + 0.25 * depth),
                                facecolor=CPK[symbols[i]], edgecolor="#2b2b2b", lw=0.8,
                                alpha=0.75 + 0.25 * depth, zorder=2 + depth))


def render_xyzrender(picked, rots, style, panel_px, fps, rotate_frames, hold_frames, end_frames, carbon_color=None):
    """Render each panel with xyzrender (style preset, e.g. 'pmol' or 'paton') frame by frame: atom types are
    taken from the current frame, bonds come from the final molecule and fade in as atoms reach bonding distance.
    Returns a list of PIL images (all panels side by side, with titles)."""
    import copy
    from io import BytesIO
    from PIL import ImageDraw, ImageFont
    from matplotlib import font_manager
    from xyzgraph import build_graph
    from xyzrender.config import build_config
    from xyzrender.export import svg_to_png_bytes
    from xyzrender.gif import _fixed_viewport
    from xyzrender.renderer import render_svg

    cfg0 = build_config(style, hy=True)  # show all hydrogens (xyzrender hides C-H by default)
    cfg0.auto_orient = False
    if carbon_color:
        from xyzrender.colors import resolve_color
        cfg0.color_overrides = {**(cfg0.color_overrides or {}), "C": resolve_color(carbon_color)}
    panels = []  # per molecule: (graph, frames as (pos, symbols, bond_opacity), final bond lengths, cfg)
    for (_, _, frames, _), r in zip(picked, rots):
        x_end, sym_end = frames[-1]
        x_end = x_end @ r
        graph = build_graph(list(zip(sym_end, [tuple(map(float, p)) for p in x_end])))
        edges = list(graph.edges())
        d_end = np.array([np.linalg.norm(x_end[i] - x_end[j]) for i, j in edges])
        seq = []
        for f, (x, sym) in enumerate(frames):  # noise -> molecule
            x = x @ r
            t = f / (len(frames) - 1)
            d = np.array([np.linalg.norm(x[i] - x[j]) for i, j in edges])
            op = t ** 3 * np.clip((1.6 * d_end - d) / (0.4 * d_end), 0, 1)
            seq.append((x, sym, op))
        seq += [seq[-1]] * hold_frames
        c = x_end.mean(0)
        for k in range(1, rotate_frames + 1):  # final molecule spins around the vertical axis
            seq.append(((x_end - c) @ rotation_y(2 * np.pi * k / rotate_frames).T + c, sym_end, np.ones(len(edges))))
        seq += [seq[-1]] * end_frames
        cfg = _fixed_viewport([{"symbols": sym_end, "positions": p} for p, _, _ in seq], cfg0,
                              rotation_axis=np.array([0.0, 1.0, 0.0]))
        panels.append((graph, edges, seq, cfg))
    span = max(cfg.fixed_span for *_, cfg in panels)  # same scale (Å per pixel) in every panel
    for *_, cfg in panels:
        cfg.fixed_span = span

    # render every panel frame first, then crop empty rows (molecules are wider than tall)
    n_panels, n_frames = len(panels), len(panels[0][2])
    tiles = [[None] * n_panels for _ in range(n_frames)]
    top, bottom = panel_px, 0
    for k, (graph, edges, seq, cfg) in enumerate(panels):
        for f, (x, sym, op) in enumerate(seq):
            for i in range(len(sym)):
                graph.nodes[i]["position"] = tuple(map(float, x[i]))
                graph.nodes[i]["symbol"] = sym[i]
            for (i, j), o in zip(edges, op):
                graph[i][j]["diffuse_opacity"] = float(o)
            png = svg_to_png_bytes(render_svg(graph, cfg, _log=False), size=panel_px)
            tile = Image.open(BytesIO(png)).convert("RGB").resize((panel_px, panel_px))
            bbox = Image.eval(tile, lambda v: 255 - v).getbbox()
            if bbox:
                top, bottom = min(top, bbox[1]), max(bottom, bbox[3])
            tiles[f][k] = tile
    pad = int(panel_px * 0.04)
    top, bottom = max(top - pad, 0), min(bottom + pad, panel_px)
    body = bottom - top

    font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), int(panel_px * 0.045))
    small = ImageFont.truetype(font_manager.findfont("DejaVu Sans Mono"), int(panel_px * 0.036))
    head, foot = int(panel_px * 0.2), int(panel_px * 0.1)
    n_traj = len(picked[0][2])
    images = []
    for f in range(n_frames):
        canvas = Image.new("RGB", (panel_px * n_panels, body + head + foot), "white")
        draw = ImageDraw.Draw(canvas)
        for k, (target, smi, _, dft) in enumerate(picked):
            canvas.paste(tiles[f][k].crop((0, top, panel_px, bottom)), (k * panel_px, head))
            cx = k * panel_px + panel_px // 2
            draw.text((cx, head * 0.72), f"target gap {target:g} eV", font=font, fill="#555555", anchor="mm")
            if f >= n_traj:
                label = smi if dft is None else f"{smi}   DFT gap {dft:.2f} eV"
                draw.text((cx, head + body + foot * 0.4), label, font=small, fill="#555555", anchor="mm")
        t = min(f / (n_traj - 1), 1.0)
        caption = f"noise \u2192 molecule    t = {t:.2f}" if f < n_traj else "generated molecules (t = 1)"
        draw.text((canvas.width // 2, head * 0.12), caption, font=font, fill="#222222", anchor="mt")
        images.append(canvas.convert("P", palette=Image.ADAPTIVE, colors=192))
    return images


def pick_trajectories(model, size_pred, fm, config, targets, device, n_try, steps):
    """For each target, sample n_try molecules and keep the first stable, valid one."""
    picked = []
    for t in targets:
        mols, traj = sample(model, size_pred, fm, config, [t] * n_try, device, steps=steps, return_traj=True)
        sizes = [len(s) for s, _ in mols]
        offsets = np.cumsum([0] + sizes)
        for k, (symbols, pos) in enumerate(mols):
            if check_stability(symbols, pos)[0] and mol_to_smiles(symbols, pos) and len(symbols) >= 12:
                sl = slice(offsets[k], offsets[k + 1])
                frames = [(x[sl].numpy(), [SYMBOLS[a] for a in h[sl].argmax(-1).tolist()]) for x, h in traj]
                picked.append((t, mol_to_smiles(symbols, pos), frames))
                break
        else:
            raise RuntimeError(f"No stable molecule among {n_try} samples for target {t} eV")
    return picked


def pick_from_dft(model, size_pred, fm, config, dft_dir, device, steps, min_atoms=12, batch_size=500):
    """Replay dft_eval.py's candidate generation and return the trajectories of the DFT-verified molecules
    closest to their targets."""
    with open(os.path.join(dft_dir, "candidates.json")) as f:
        meta = json.load(f)
    seed, n_generate = meta.get("seed", 0), meta.get("n_generate", 3000)
    done = {}
    for path in glob.glob(os.path.join(dft_dir, "molecules", "*.json")):
        with open(path) as f:
            r = json.load(f)
        if "error" not in r and r["dft_opt"]["scf_converged"]:
            done[r["id"]] = r
    chosen = {}
    for t in meta["targets"]:
        ok = [r for r in done.values() if r["target_ev"] == t and len(r["symbols"]) >= min_atoms
              and not (r["oo_bond"] or r["graph_changed_conformation"] or r["graph_changed_relaxation"])
              and r.get("smiles_opt") == r["smiles"]]
        if not ok:
            raise RuntimeError(f"no finished DFT molecule for target {t} eV")
        chosen[t] = min(ok, key=lambda r: abs(r["dft_opt"]["gap_ev"] - t))

    torch.manual_seed(seed)  # same RNG stream as dft_eval.build_candidates
    picked = []
    for t in meta["targets"]:
        target = np.array(chosen[t]["pos"])
        found = None
        for i in range(0, n_generate, batch_size):
            mols, traj = sample(model, size_pred, fm, config, [t] * min(batch_size, n_generate - i), device,
                                steps=steps, return_traj=True)
            if found is not None:
                continue  # keep consuming the RNG exactly like the original run
            offsets = np.cumsum([0] + [len(s) for s, _ in mols])
            for k, (symbols, pos) in enumerate(mols):
                if symbols == chosen[t]["symbols"] and np.abs(np.asarray(pos) - target).max() < 1e-3:
                    sl = slice(offsets[k], offsets[k + 1])
                    found = [(x[sl].numpy(), [SYMBOLS[a] for a in h[sl].argmax(-1).tolist()]) for x, h in traj]
                    break
        if found is None:
            raise RuntimeError(f"could not reproduce molecule {chosen[t]['id']} (different GPU/seed?)")
        r = chosen[t]
        print(f"  {r['id']}: {r['smiles']}  DFT gap {r['dft_opt']['gap_ev']:.2f} eV (target {t:g})")
        picked.append((t, r["smiles"], found, r["dft_opt"]["gap_ev"]))
    return picked


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", default=None)
    p.add_argument("--targets", type=float, nargs="+", default=[4.2, 6.8, 9.4])
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--n_try", type=int, default=64)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--rotate_frames", type=int, default=72)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--from_dft", default="results/dft",
                   help="show DFT-verified molecules from this dft_eval.py output dir ('' = sample fresh ones)")
    p.add_argument("--style", default="pmol",
                   help="xyzrender preset (pmol, paton, default, flat, tube, ...) or 'matplotlib' for the built-in renderer")
    p.add_argument("--carbon_color", default="#aaaaaa",
                   help="carbon colour override ('' keeps the preset's; #aaaaaa = xyzrender default style)")
    p.add_argument("--panel_px", type=int, default=360)
    p.add_argument("--out", default="assets/generation.gif")
    args = p.parse_args()

    plot_style.apply()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, size_pred, fm, config = load_run(args.run_dir, device)
    if args.from_dft and os.path.exists(os.path.join(args.from_dft, "candidates.json")):
        picked = pick_from_dft(model, size_pred, fm, config, args.from_dft, device, args.steps)
    else:
        picked = [(t, smi, fr, None) for t, smi, fr in
                  pick_trajectories(model, size_pred, fm, config, args.targets, device, args.n_try, args.steps)]

    # orient each molecule by its final geometry; the same rotation is applied to every frame
    rots = [principal_axes(frames[-1][0]) for _, _, frames, _ in picked]
    extent = max(np.abs(frames[-1][0] @ r).max() for (_, _, frames, _), r in zip(picked, rots)) + 0.8

    if args.style != "matplotlib":
        images = render_xyzrender(picked, rots, args.style, args.panel_px, args.fps, args.rotate_frames,
                                  hold_frames=args.fps // 2, end_frames=args.fps, carbon_color=args.carbon_color)
        save_gif(images, args.out, args.fps, picked)
        return

    fig, axes = plt.subplots(1, len(picked), figsize=(3.4 * len(picked), 3.9), facecolor="white")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.83, bottom=0.04, wspace=0.02)
    title = fig.suptitle("", fontsize=12, color=plot_style.TEXT)
    images = []

    def snap():
        fig.canvas.draw()
        images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]).convert(
            "P", palette=Image.ADAPTIVE, colors=128))

    n_frames = len(picked[0][2])
    for f in range(n_frames):
        t = f / (n_frames - 1)
        for ax, (target, smi, frames, _), r in zip(axes, picked, rots):
            x, symbols = frames[f]
            draw_molecule(ax, symbols, x @ r, bond_alpha=t ** 3, extent=extent)
            ax.set_title(f"target gap {target:g} eV", fontsize=10, color=plot_style.TEXT_SECONDARY)
        title.set_text(f"noise → molecule      t = {t:.2f}")
        snap()
    hold = images[-1]
    images += [hold] * (args.fps // 2)
    for k in range(1, args.rotate_frames + 1):
        rot = rotation_y(2 * np.pi * k / args.rotate_frames)
        for ax, (target, smi, frames, dft), r in zip(axes, picked, rots):
            x, symbols = frames[-1]
            draw_molecule(ax, symbols, x @ r @ rot.T, bond_alpha=1.0, extent=extent)
            ax.set_title(f"target gap {target:g} eV", fontsize=10, color=plot_style.TEXT_SECONDARY)
            label = smi if dft is None else f"{smi}\nDFT gap {dft:.2f} eV"
            ax.text(0, -extent + 0.15, label, ha="center", va="bottom", fontsize=8, linespacing=1.5,
                    color=plot_style.TEXT_SECONDARY, family="monospace")
        title.set_text("generated molecules (t = 1)")
        snap()
    images += [images[-1]] * args.fps

    save_gif(images, args.out, args.fps, picked)


def save_gif(images, out, fps, picked):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    images[0].save(out, save_all=True, append_images=images[1:], duration=[1000 // fps] * len(images),
                   loop=0, optimize=True, disposal=2)
    print(f"Saved {out} ({len(images)} frames, {os.path.getsize(out) / 1e6:.1f} MB)")
    for target, smi, _, dft in picked:
        print(f"  target {target:g} eV: {smi}" + ("" if dft is None else f" (DFT gap {dft:.2f} eV)"))


if __name__ == "__main__":
    main()
