"""Download and extract the QM9 .xyz files (Ramakrishnan et al., 2014) into data/qm9/raw.

Figshare sometimes blocks scripted downloads. If that happens, download dsgdb9nsd.xyz.tar.bz2 manually from
https://doi.org/10.6084/m9.figshare.c.978904.v5 ("Data for 133885 GDB-9 molecules") and run
    python download_dataset.py --archive path/to/dsgdb9nsd.xyz.tar.bz2
"""
import os
import sys
import tarfile
import argparse
import urllib.request

URL = "https://springernature.figshare.com/ndownloader/files/3195389"
N_MOLECULES = 133885


def extract(archive, raw_dir):
    print(f"Extracting {archive} -> {raw_dir}")
    with tarfile.open(archive, "r:bz2") as tar:
        tar.extractall(raw_dir, filter="data")
    n = len([f for f in os.listdir(raw_dir) if f.endswith(".xyz")])
    print(f"{n} .xyz files in {raw_dir}" + ("" if n == N_MOLECULES else f" (expected {N_MOLECULES})"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data/qm9")
    p.add_argument("--archive", default=None, help="use an already downloaded dsgdb9nsd.xyz.tar.bz2")
    args = p.parse_args()

    raw_dir = os.path.join(args.root, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    if len(os.listdir(raw_dir)) >= N_MOLECULES:
        print(f"QM9 already present in {raw_dir}")
        return

    archive = args.archive or os.path.join(args.root, "dsgdb9nsd.xyz.tar.bz2")
    if not os.path.exists(archive):
        print(f"Downloading QM9 from {URL} ...")
        try:
            req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as r, open(archive, "wb") as f:
                if r.status != 200 or int(r.headers.get("Content-Length") or 1) == 0:
                    raise RuntimeError(f"HTTP {r.status}")
                while chunk := r.read(1 << 20):
                    f.write(chunk)
        except Exception as e:
            if os.path.exists(archive):
                os.remove(archive)
            sys.exit(f"Automatic download failed ({e}).\n{__doc__}")
    extract(archive, raw_dir)
    print("Done. The dataset is processed on first use (python data.py).")


if __name__ == "__main__":
    main()
