"""Cek integritas data: data/raw identik dengan audit (MD5, read-only) dan data/split tetap symlink.

Contoh:
  python scripts/verify_data.py
Keluar dengan kode 1 bila ada masalah (jalankan 02_split.py untuk membangun ulang data/split).
"""
import hashlib
import sys

import pandas as pd

from common import ROOT


def main():
    a = pd.read_csv(ROOT / "reports/audit_files.csv", keep_default_na=False)
    raw = ROOT / "data/raw"
    bad = [r for r, m in zip(a.relpath, a.md5) if hashlib.md5((raw / r).read_bytes()).hexdigest() != m]
    not_ro = [p for p in raw.rglob("*") if p.is_file() and not (p.stat().st_file_attributes & 1)] \
        if sys.platform == "win32" else []
    split = ROOT / "data/split"
    non_link = [p for p in split.rglob("*") if p.is_file() and not p.is_symlink() and p.suffix != ".cache"]
    print(f"data/raw  : {len(a)} file, MD5 berbeda: {len(bad)}, tidak read-only: {len(not_ro)}")
    print(f"data/split: file bukan symlink (selain .cache): {len(non_link)}")
    for p in (bad + [str(x) for x in non_link])[:10]:
        print("  -", p)
    if bad or not_ro or non_link:
        print("MASALAH ditemukan." + (" Bangun ulang split: python scripts/02_split.py" if non_link and not bad else ""))
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
