"""Tahap 1: audit dataset (read-only terhadap data/raw).

Output:
  reports/audit_files.csv        satu baris per file (md5, format, mode, ukuran, statistik warna, pHash)
  reports/near_dup_pairs.csv     pasangan dengan jarak pHash <= --max-pair-dist (invarian rotasi/flip)
  reports/dup_groups.csv         grup duplikat (exact + near pada --phash-threshold), union-find
  reports/samples_<kelas>.png    grid sampel acak per kelas
  reports/dup_pairs_*.png        grid visual pasangan duplikat per rentang jarak & lintas kelas
  reports/audit.md               ringkasan
"""
import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imagehash
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

from common import IMG_EXTS, ROOT, SEED, list_images, md_table, seed_everything  # noqa: E402

# 8 transformasi dihedral: identitas, rotasi 90/180/270, dan versi flip-nya
DIHEDRAL = [
    ("id", lambda im: im),
    ("rot90", lambda im: im.transpose(Image.Transpose.ROTATE_90)),
    ("rot180", lambda im: im.transpose(Image.Transpose.ROTATE_180)),
    ("rot270", lambda im: im.transpose(Image.Transpose.ROTATE_270)),
    ("flipH", lambda im: im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)),
    ("flipV", lambda im: im.transpose(Image.Transpose.FLIP_TOP_BOTTOM)),
    ("transpose", lambda im: im.transpose(Image.Transpose.TRANSPOSE)),
    ("transverse", lambda im: im.transpose(Image.Transpose.TRANSVERSE)),
]


def analyze_file(args):
    path, cls, raw_root = args
    rec = {"relpath": path.relative_to(raw_root).as_posix(), "class": cls, "filename": path.name,
           "ext": path.suffix.lower(), "bytes": path.stat().st_size, "ok": False, "error": ""}
    data = path.read_bytes()
    rec["md5"] = hashlib.md5(data).hexdigest()
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            rec.update(format=im.format, mode=im.mode, width=im.width, height=im.height,
                       exif_orientation=im.getexif().get(0x0112, 0))
            im.load()  # menangkap file terpotong
            has_alpha = im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info
            rec["alpha_used"] = bool(has_alpha and np.asarray(im.convert("RGBA"))[..., 3].min() < 255)
            rgb = im.convert("RGB")
        thumb = rgb.copy()
        thumb.thumbnail((256, 256), Image.Resampling.BILINEAR)
        a = np.asarray(thumb, dtype=np.float32)
        rec.update(mean_r=a[..., 0].mean(), mean_g=a[..., 1].mean(), mean_b=a[..., 2].mean(),
                   brightness=a.mean(), contrast=a.mean(axis=2).std(),
                   frac_white=float((a.min(axis=2) > 235).mean()),
                   frac_black=float((a.max(axis=2) < 20).mean()))
        for name, fn in DIHEDRAL:
            rec[f"phash_{name}"] = str(imagehash.phash(fn(thumb)))
        rec["ok"] = True
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def source_style(r):
    """Kelompok 'gaya/sub-sumber' berdasarkan format & resolusi (indikasi asal pengambilan)."""
    if r["ext"] == ".png":
        w = int(r["width"])
        return f"PNG-{'256' if w == 256 else ('1000' if w == 1000 else 'lain')}-{r['mode']}"
    if max(r["width"], r["height"]) >= 3000:
        return "JPG-foto>=3000"
    if 640 in (r["width"], r["height"]):
        return "JPG-640"
    return "JPG-lain"


def hex_to_u64(series):
    return np.array([int(h, 16) for h in series], dtype=np.uint64)


def pairwise_min_dist(df, max_dist, chunk=256):
    """Jarak Hamming minimum antar file, diminimumkan atas 8 transformasi dihedral."""
    h0 = hex_to_u64(df["phash_id"])
    hk = np.stack([hex_to_u64(df[f"phash_{n}"]) for n, _ in DIHEDRAL])  # (8, N)
    n = len(h0)
    nn_dist = np.full(n, 64, dtype=np.int16)
    rows = []
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d = np.bitwise_count(h0[s:e][None, :, None] ^ hk[:, None, :])  # (8, c, N)
        best_k = d.argmin(axis=0)
        dmin = d.min(axis=0).astype(np.int16)
        idx = np.arange(s, e)
        dmin[idx - s, idx] = 99  # diri sendiri
        nn_dist[s:e] = dmin.min(axis=1)
        ii, jj = np.nonzero(dmin <= max_dist)
        keep = (ii + s) < jj
        for i, j in zip(ii[keep], jj[keep]):
            rows.append((i + s, j, int(dmin[i, j]), DIHEDRAL[best_k[i, j]][0]))
    pairs = pd.DataFrame(rows, columns=["i", "j", "dist", "xform"])
    return pairs, nn_dist


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def sample_grid(df_cls, raw_root, out_path, n, rng, title):
    take = df_cls.sample(n=min(n, len(df_cls)), random_state=rng)
    cols = 4
    rows = int(np.ceil(len(take) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.3))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, r in zip(np.ravel(axes), take.itertuples()):
        with Image.open(raw_root / r.relpath) as im:
            im = im.convert("RGB")
            im.thumbnail((300, 300))
            ax.imshow(im)
        ax.set_title(f"{r.filename}\n{r.width}x{r.height} {r.mode}", fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90)
    plt.close(fig)


def pair_grid(pairs, df, raw_root, out_path, title, max_pairs=12):
    pairs = pairs.head(max_pairs)
    if pairs.empty:
        return False
    cols = 4  # 2 pasangan per baris
    rows = int(np.ceil(len(pairs) / 2))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.8, rows * 3.1), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for k, p in enumerate(pairs.itertuples()):
        for m, idx in enumerate((p.i, p.j)):
            r = df.iloc[idx]
            ax = axes[k // 2, (k % 2) * 2 + m]
            with Image.open(raw_root / r.relpath) as im:
                im = im.convert("RGB")
                im.thumbnail((240, 240))
                ax.imshow(im)
            lbl = f"{r['class']}\n{r.filename}"
            if m == 1:
                lbl += f"\nd={p.dist} ({p.xform})"
            ax.set_title(lbl, fontsize=6)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=ROOT / "data/raw")
    ap.add_argument("--reports", type=Path, default=ROOT / "reports")
    ap.add_argument("--expected", type=str, default=str(ROOT / "configs/dataset.json"),
                    help="JSON berisi jumlah per kelas menurut deskripsi dataset")
    ap.add_argument("--phash-threshold", type=int, default=6,
                    help="Hamming distance <= ini dianggap near-duplicate (pHash 64-bit)")
    ap.add_argument("--max-pair-dist", type=int, default=12, help="simpan pasangan sampai jarak ini (analisis)")
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--reuse", action="store_true", help="pakai reports/audit_files.csv yang sudah ada (tanpa hash ulang)")
    args = ap.parse_args()
    seed_everything(args.seed)
    rep = args.reports
    rep.mkdir(parents=True, exist_ok=True)

    # ---------------- 1. struktur & inventaris ----------------
    class_dirs = sorted(d for d in args.raw.iterdir() if d.is_dir())
    tasks, non_image = [], []
    for d in class_dirs:
        for p in list_images(d):
            (tasks if p.suffix.lower() in IMG_EXTS else non_image).append((p, d.name, args.raw))
    stray = [p.name for p in args.raw.iterdir() if p.is_file()]
    print(f"{len(class_dirs)} folder kelas, {len(tasks)} file gambar, {len(non_image)} non-gambar")

    cache = rep / "audit_files.csv"
    if args.reuse and cache.exists():
        df = pd.read_csv(cache, keep_default_na=False, dtype={"error": str})
        df["ok"] = df["ok"].astype(str) == "True"
        print(f"memakai cache {cache}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            recs = list(ex.map(analyze_file, tasks, chunksize=32))
        df = pd.DataFrame(recs).sort_values("relpath").reset_index(drop=True)
        df.to_csv(cache, index=False)
    for c in ("width", "height", "exif_orientation"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["alpha_used"] = df["alpha_used"].astype(str) == "True"
    bad = df[~df["ok"]]
    good = df[df["ok"]].reset_index(drop=True)

    # ---------------- 2. jumlah vs deskripsi ----------------
    expected = json.loads(Path(args.expected).read_text(encoding="utf-8")) if Path(args.expected).exists() else {}
    exp_cls = expected.get("classes", {})
    counts = df.groupby("class").size().rename("jumlah_file").reset_index()
    counts["deskripsi"] = counts["class"].map(lambda c: exp_cls.get(c, "?"))
    counts["selisih"] = [(a - b) if isinstance(b, int) else "?" for a, b in zip(counts.jumlah_file, counts.deskripsi)]

    # ---------------- 3. format, mode, resolusi ----------------
    fmt_tab = pd.crosstab(df["class"], df["ext"]).reset_index()
    mode_tab = pd.crosstab(good["class"], good["mode"]).reset_index()
    good["res"] = good["width"].astype(str) + "x" + good["height"].astype(str)
    res_tab = good.groupby("res").size().sort_values(ascending=False).rename("n").reset_index()
    res_cls = pd.crosstab(good["class"], good["res"].where(good["res"].isin(res_tab.res.head(6)), "lainnya")).reset_index()
    good["style"] = good.apply(source_style, axis=1)
    style_tab = pd.crosstab(good["style"], good["class"], margins=True, margins_name="Total").reset_index()
    color_tab = good.groupby("class")[["brightness", "contrast", "mean_r", "mean_g", "mean_b", "frac_white", "frac_black"]] \
        .mean().round(3).reset_index()
    size_tab = df.groupby(["class", "ext"])["bytes"].median().div(1024).round(0).rename("median_KB").reset_index()

    # stem angka sama dengan ekstensi berbeda di kelas yang sama
    good["stem"] = good["filename"].str.rsplit(".", n=1).str[0]
    stem_multi = good.groupby(["class", "stem"]).size()
    stem_multi = stem_multi[stem_multi > 1]
    odd_names = good[~good["stem"].str.match(r"^[A-Za-z _]+_\d+$")]

    # ---------------- 4. duplikat ----------------
    md5_counts = good.groupby("md5")["class"].agg(["size", "nunique"])
    exact_md5 = md5_counts[md5_counts["size"] > 1]
    exact_files = good[good.md5.isin(exact_md5.index)]
    n_exact_redundant = int((exact_md5["size"] - 1).sum())
    exact_cross = exact_md5[exact_md5["nunique"] > 1]

    pairs, nn_dist = pairwise_min_dist(good, args.max_pair_dist)
    good["nn_dist"] = nn_dist
    pairs["class_i"] = good["class"].values[pairs.i]
    pairs["class_j"] = good["class"].values[pairs.j]
    pairs["file_i"] = good["relpath"].values[pairs.i]
    pairs["file_j"] = good["relpath"].values[pairs.j]
    pairs["same_md5"] = good["md5"].values[pairs.i] == good["md5"].values[pairs.j]
    pairs["cross_class"] = pairs.class_i != pairs.class_j
    pairs.sort_values(["dist", "i", "j"]).to_csv(rep / "near_dup_pairs.csv", index=False)

    # sensitivitas threshold
    sens = []
    for t in range(0, args.max_pair_dist + 1):
        p = pairs[pairs.dist <= t]
        uf = UnionFind(len(good))
        for i, j in zip(p.i, p.j):
            uf.union(i, j)
        roots = pd.Series([uf.find(k) for k in range(len(good))])
        gs = roots.value_counts()
        sens.append({"threshold": t, "pasangan": len(p), "pasangan_lintas_kelas": int(p.cross_class.sum()),
                     "pasangan_via_rotasi_flip": int((p.xform != "id").sum()),
                     "file_dalam_grup": int(gs[gs > 1].sum()), "grup": int((gs > 1).sum()),
                     "file_redundan": int((gs[gs > 1] - 1).sum())})
    sens = pd.DataFrame(sens)

    # grup final pada threshold terpilih (exact md5 selalu masuk, d=0 atau via md5)
    t = args.phash_threshold
    uf = UnionFind(len(good))
    for i, j in zip(pairs.i[pairs.dist <= t], pairs.j[pairs.dist <= t]):
        uf.union(i, j)
    for _, idx in good.groupby("md5").groups.items():
        idx = list(idx)
        for k in idx[1:]:
            uf.union(idx[0], k)
    good["group_root"] = [uf.find(k) for k in range(len(good))]
    gsize = good.groupby("group_root")["relpath"].transform("size")
    gcls = good.groupby("group_root")["class"].transform(lambda s: "|".join(sorted(set(s))))
    good["group_size"] = gsize
    good["group_classes"] = gcls
    good["group_id"] = good["group_root"].rank(method="dense").astype(int) - 1
    good[["relpath", "class", "md5", "style", "nn_dist"]].to_csv(rep / "audit_style.csv", index=False)
    dups = good[good.group_size > 1][["group_id", "relpath", "class", "md5", "group_size", "group_classes", "nn_dist"]]
    dups.sort_values(["group_id", "relpath"]).to_csv(rep / "dup_groups.csv", index=False)
    grp = good[good.group_size > 1].groupby("group_root").agg(size=("relpath", "size"), ncls=("class", "nunique"))
    cross_groups = grp[grp.ncls > 1]

    per_class_dup = good.assign(in_group=good.group_size > 1).groupby("class").agg(
        file=("relpath", "size"), dalam_grup_dup=("in_group", "sum"),
        nn_dist_median=("nn_dist", "median")).reset_index()
    per_class_dup["unik_setelah_dedup"] = good.groupby("class")["group_root"].nunique().values

    # ---------------- 5. visual ----------------
    rng = np.random.RandomState(args.seed)
    for c in sorted(good["class"].unique()):
        sample_grid(good[good["class"] == c], args.raw, rep / f"samples_{c}.png", args.samples, rng,
                    f"{c} — {args.samples} sampel acak (seed {args.seed})")
    pr = pairs.copy()
    bands = [(0, 0), (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]
    made = []
    for lo, hi in bands:
        sub = pr[(pr.dist >= lo) & (pr.dist <= hi)].sample(frac=1, random_state=args.seed)
        if pair_grid(sub, good, args.raw, rep / f"dup_pairs_d{lo:02d}-{hi:02d}.png",
                     f"Pasangan pHash jarak {lo}-{hi} (acak, maks 12)"):
            made.append(f"dup_pairs_d{lo:02d}-{hi:02d}.png")
    cross = pr[pr.cross_class].sort_values("dist")
    if pair_grid(cross, good, args.raw, rep / "dup_pairs_cross_class.png",
                 "Pasangan LINTAS KELAS terdekat", max_pairs=16):
        made.append("dup_pairs_cross_class.png")
    rot = pr[(pr.xform != "id") & (pr.dist <= t)].sample(frac=1, random_state=args.seed)
    if pair_grid(rot, good, args.raw, rep / "dup_pairs_rotflip.png", f"Duplikat via rotasi/flip (d<={t})"):
        made.append("dup_pairs_rotflip.png")

    # histogram jarak tetangga terdekat
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(nn_dist, bins=np.arange(0, 34) - 0.5, color="#4C72B0")
    ax.axvline(t + 0.5, color="red", ls="--", label=f"threshold={t}")
    ax.set_xlabel("Jarak pHash ke tetangga terdekat (min atas 8 rotasi/flip)")
    ax.set_ylabel("Jumlah gambar")
    ax.legend()
    fig.tight_layout()
    fig.savefig(rep / "nn_dist_hist.png", dpi=100)
    plt.close(fig)

    # ---------------- 6. laporan ----------------
    L = ["# Audit Dataset — Chilli Leaf Disease (Mendeley tm3v4zmh7c)", "",
         f"- Root data: `{args.raw.relative_to(ROOT).as_posix()}` (read-only)",
         f"- Seed: {args.seed}; pHash 64-bit (hash_size=8) dihitung pada thumbnail 256 px, "
         f"jarak = Hamming minimum atas 8 transformasi rotasi/flip; threshold near-duplicate = **{t}**", "",
         "## 1. Struktur & jumlah per kelas", "",
         f"Folder kelas: {', '.join(d.name for d in class_dirs)}. "
         f"File non-gambar: {len(non_image)}. File lepas di root: {stray or 'tidak ada'}.", "",
         md_table(counts), "",
         f"Total: **{len(df)}** file; deskripsi: **{expected.get('total', '?')}**.", "",
         "## 2. File rusak / tidak terbaca", "",
         f"{len(bad)} file." + ("" if bad.empty else "\n\n" + md_table(bad[["relpath", "error"]])), "",
         "## 3. Format, mode warna, resolusi", "", "### Ekstensi per kelas", "", md_table(fmt_tab), "",
         "### Mode warna per kelas", "", md_table(mode_tab), "",
         f"File dengan kanal alpha yang benar-benar transparan: {int(good.alpha_used.sum())}. "
         f"File dengan tag EXIF orientation != 0/1: {int((good.exif_orientation > 1).sum())}.", "",
         "### Resolusi", "", md_table(res_tab.head(10)), "", md_table(res_cls), "",
         "### Gaya / sub-sumber (format + resolusi) per kelas", "",
         "Distribusi gaya yang berbeda antar kelas = risiko *shortcut learning* (model belajar sumber, bukan penyakit).", "",
         md_table(style_tab), "",
         "### Ukuran file median (KB) per kelas & ekstensi", "", md_table(size_tab), "",
         "### Statistik warna rata-rata per kelas (thumbnail 256 px, skala 0–255)", "",
         "`frac_white` = fraksi piksel hampir putih (min RGB > 235), `frac_black` = hampir hitam (max RGB < 20).", "",
         md_table(color_tab), "",
         "### Nama file tidak standar", "",
         (md_table(odd_names[["relpath", "md5", "width", "height"]]) if len(odd_names) else "Tidak ada."), "",
         f"Stem nama sama dengan ekstensi berbeda dalam satu kelas: {len(stem_multi)}.", "",
         "## 4. Duplikat", "", "### Exact (MD5)", "",
         f"- Grup MD5 dengan >1 file: **{len(exact_md5)}** ({len(exact_files)} file, {n_exact_redundant} redundan)",
         f"- Grup exact **lintas kelas**: **{len(exact_cross)}**", "",
         "### Near-duplicate (pHash, invarian rotasi/flip) — sensitivitas threshold", "",
         "`file_redundan` = file yang akan terbuang jika tiap grup disisakan 1.", "",
         md_table(sens), "",
         f"### Grup pada threshold terpilih ({t}, termasuk exact)", "",
         f"- Grup: **{len(grp)}**, file di dalam grup: **{int(grp['size'].sum()) if len(grp) else 0}**, "
         f"ukuran grup maks: {int(grp['size'].max()) if len(grp) else 0}",
         f"- Grup **lintas kelas**: **{len(cross_groups)}** "
         f"({int(cross_groups['size'].sum()) if len(cross_groups) else 0} file)", "",
         md_table(per_class_dup), "",
         "Detail: `reports/dup_groups.csv`, `reports/near_dup_pairs.csv`. "
         "Distribusi jarak tetangga terdekat: `reports/nn_dist_hist.png`.", "",
         "## 5. Visual", "",
         "Grid sampel: " + ", ".join(f"`samples_{c}.png`" for c in sorted(good['class'].unique())), "",
         "Grid pasangan duplikat: " + ", ".join(f"`{m}`" for m in made), ""]
    (rep / "audit.md").write_text("\n".join(L), encoding="utf-8")

    summary = {"total": len(df), "bad": len(bad), "exact_groups": len(exact_md5), "exact_redundant": n_exact_redundant,
               "exact_cross": len(exact_cross), "threshold": t, "near_groups": len(grp),
               "near_cross_groups": len(cross_groups)}
    (rep / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
