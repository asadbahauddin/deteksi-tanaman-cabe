"""Tahap 2: dedup + group-aware stratified split 70/15/15.

Output:
  splits/duplicates.csv            tindakan per file (kept / removed_*)
  splits/{train,val,test}.csv      path, label, md5, group_id, style
  data/split/{train,val,test}/<kelas>/   symlink (jika didukung) atau copy
  reports/split.md                 tabel jumlah + hasil verifikasi
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold

from common import ROOT, md_table, seed_everything

SPLITS = ("train", "val", "test")


class UnionFind:
    def __init__(self, items):
        self.p = {x: x for x in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def link_or_copy(src: Path, dst: Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode in ("auto", "symlink"):
        try:
            os.symlink(src.resolve(), dst)
            return "symlink"
        except OSError:
            if mode == "symlink":
                raise
    shutil.copy2(src, dst)
    os.chmod(dst, 0o666)  # salinan tidak perlu read-only
    return "copy"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/split.yaml")
    ap.add_argument("--no-materialize", action="store_true", help="hanya CSV, tanpa membangun data/split")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed_everything(cfg["seed"])
    P = lambda k: ROOT / cfg[k]  # noqa: E731

    df = pd.read_csv(P("audit_files"), keep_default_na=False)
    df = df[df["ok"].astype(str) == "True"].merge(
        pd.read_csv(P("audit_style"))[["relpath", "style"]], on="relpath", how="left")
    df = df.sort_values("relpath").reset_index(drop=True)
    n0 = len(df)

    # ---------------- 1. buang file sesuai keputusan ----------------
    action = pd.Series("kept", index=df.index)
    kept_as = pd.Series("", index=df.index)
    for reason, files in cfg["drop_files"].items():
        missing = set(files) - set(df.relpath)
        if missing:
            raise SystemExit(f"File di drop_files tidak ditemukan: {missing}")
        action[df.relpath.isin(files)] = f"removed_{reason}"
    if cfg["drop_exact_duplicates"]:
        alive = action == "kept"
        first = df[alive].groupby("md5")["relpath"].transform("first")
        is_dup = df[alive].relpath != first
        action[is_dup[is_dup].index] = "removed_exact_dup"
        kept_as[is_dup[is_dup].index] = first[is_dup]
    df["action"], df["kept_as"] = action, kept_as

    # ---------------- 2. grup near-duplicate ----------------
    keep = df[df.action == "kept"].copy()
    uf = UnionFind(keep.relpath)
    pairs = pd.read_csv(P("near_pairs"))
    pairs = pairs[pairs.dist <= cfg["group_phash_threshold"]]
    alive = set(keep.relpath)
    for a, b in zip(pairs.file_i, pairs.file_j):
        if a in alive and b in alive:
            uf.union(a, b)
    keep["root"] = keep.relpath.map(uf.find)
    keep["group_id"] = keep["root"].rank(method="dense").astype(int) - 1
    df = df.merge(keep[["relpath", "group_id"]], on="relpath", how="left")
    df["group_id"] = df["group_id"].astype("Int64")

    # ---------------- 3. stratified group split ----------------
    strat = keep[cfg["stratify_by"]].astype(str).agg("|".join, axis=1)
    sgkf = StratifiedGroupKFold(n_splits=cfg["n_folds"], shuffle=True, random_state=cfg["seed"])
    fold = np.empty(len(keep), dtype=int)
    for k, (_, te) in enumerate(sgkf.split(keep, strat, groups=keep.group_id)):
        fold[te] = k
    r = cfg["ratios"]
    n_test = round(cfg["n_folds"] * r["test"])
    n_val = round(cfg["n_folds"] * r["val"])
    keep["split"] = np.where(fold < n_test, "test", np.where(fold < n_test + n_val, "val", "train"))

    out = P("out_csv_dir")
    out.mkdir(parents=True, exist_ok=True)
    raw_rel = Path(cfg["raw_dir"])
    for s in SPLITS:
        sub = keep[keep.split == s].sort_values("relpath")
        pd.DataFrame({"path": [(raw_rel / p).as_posix() for p in sub.relpath], "label": sub["class"],
                      "md5": sub.md5, "group_id": sub.group_id, "style": sub["style"]}) \
            .to_csv(out / f"{s}.csv", index=False)
    df = df.merge(keep[["relpath", "split"]], on="relpath", how="left")
    df[["relpath", "class", "md5", "action", "kept_as", "group_id", "split"]].to_csv(out / "duplicates.csv", index=False)

    # ---------------- 4. verifikasi ----------------
    sets = {s: keep[keep.split == s] for s in SPLITS}
    checks = {}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            checks[f"md5 {a}∩{b}"] = len(set(sets[a].md5) & set(sets[b].md5))
            checks[f"group {a}∩{b}"] = len(set(sets[a].group_id) & set(sets[b].group_id))
    split_of = dict(zip(keep.relpath, keep.split))
    allp = pd.read_csv(P("near_pairs"))
    allp = allp[allp.file_i.isin(split_of) & allp.file_j.isin(split_of)]
    cross = allp[allp.file_i.map(split_of) != allp.file_j.map(split_of)]
    checks[f"pasangan pHash<={cfg['group_phash_threshold']} lintas split"] = int(
        (cross.dist <= cfg["group_phash_threshold"]).sum())
    cross_dist = cross.groupby("dist").size().rename("pasangan_lintas_split").reset_index()
    ok = all(v == 0 for v in checks.values())

    # ---------------- 5. materialisasi folder ----------------
    modes = {}
    if not args.no_materialize:
        img_root = P("out_img_dir")
        if img_root.exists():
            shutil.rmtree(img_root)
        for rel, cls, s in zip(keep.relpath, keep["class"], keep.split):
            m = link_or_copy(ROOT / raw_rel / rel, img_root / s / cls / Path(rel).name, cfg["link_mode"])
            modes[m] = modes.get(m, 0) + 1
        n_disk = {s: sum(1 for p in (img_root / s).rglob("*") if p.is_file() or p.is_symlink()) for s in SPLITS}
        checks.update({f"file di data/split/{s} == csv": int(n_disk[s] != len(sets[s])) for s in SPLITS})
        ok = all(v == 0 for v in checks.values())

    # ---------------- 6. laporan ----------------
    tab = pd.crosstab(keep["class"], keep["split"])[list(SPLITS)]
    tab["total"] = tab.sum(axis=1)
    tab.loc["TOTAL"] = tab.sum()
    pct = tab[list(SPLITS)].div(tab["total"], axis=0).mul(100).round(1)
    tab_md = tab.astype(str) + pd.concat([pct.astype(str).radd(" (").add("%)"), pd.Series("", index=tab.index, name="total")], axis=1)
    groups_tab = keep.groupby("split")["group_id"].nunique().reindex(SPLITS).rename("grup").reset_index()
    style_tab = pd.crosstab([keep["class"], keep["style"]], keep["split"])[list(SPLITS)].reset_index()
    act = df.groupby(["action", "class"]).size().unstack(fill_value=0).reset_index()

    L = ["# Split dataset (Tahap 2)", "",
         f"- Config: `{args.config.relative_to(ROOT).as_posix()}`; seed {cfg['seed']}; "
         f"StratifiedGroupKFold {cfg['n_folds']} fold (stratifikasi: {' × '.join(cfg['stratify_by'])}); "
         f"grup = pHash ≤ {cfg['group_phash_threshold']} (invarian rotasi/flip) + MD5",
         f"- File awal (terbaca): {n0}; dipakai: **{len(keep)}** dalam **{keep.group_id.nunique()}** grup",
         f"- Materialisasi: {modes or 'dilewati'}", "",
         "## Tindakan per file", "", md_table(act), "",
         "## Jumlah per kelas per split", "", md_table(tab_md.reset_index()), "", md_table(groups_tab), "",
         "## Kelas × gaya per split", "", md_table(style_tab), "",
         "## Verifikasi tanpa irisan", "", md_table(pd.DataFrame(checks.items(), columns=["cek", "jumlah_irisan"])), "",
         f"**Status: {'LULUS' if ok else 'GAGAL'}**", "",
         "Pasangan near-duplicate yang berada di split berbeda (di atas threshold, untuk informasi):", "",
         md_table(cross_dist.head(8)), ""]
    (ROOT / "reports/split.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    if not ok:
        raise SystemExit("VERIFIKASI GAGAL")


if __name__ == "__main__":
    main()
