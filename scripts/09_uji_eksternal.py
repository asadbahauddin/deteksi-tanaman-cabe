"""Uji model pada gambar dari luar dataset (mis. unduhan internet) — folder per kelas.

Struktur folder (default data/uji_internet/):
  <Kelas>/            gambar berlabel (nama folder = nama kelas dataset) -> dihitung akurasi, macro-F1, confusion matrix
  _belum_dilabeli/    hanya prediksi
  _bukan_daun/        gambar bukan daun -> distribusi confidence (model closed-set selalu memilih 1 dari 6 kelas)

Cek kebocoran: setiap gambar dibandingkan dengan seluruh gambar dataset Mendeley (MD5 + pHash minimum atas
8 rotasi/flip, sama seperti 01_audit). Gambar dengan jarak <= --dup-threshold ditandai `dari_dataset`
dan metrik dilaporkan dengan & tanpa gambar tersebut.

Contoh:
  python scripts/09_uji_eksternal.py
  python scripts/09_uji_eksternal.py --folder data/uji_internet --models mobilenetv3_large_100_augB yolo11n_cls
"""
import argparse
import datetime as dt
import hashlib
import importlib
from pathlib import Path

import imagehash
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, f1_score  # noqa: E402

from common import IMG_EXTS, ROOT, md_table  # noqa: E402
from data_utils import center_224, load_rgb  # noqa: E402
from models_io import load_classifier  # noqa: E402

audit = importlib.import_module("01_audit")
ev = importlib.import_module("04_evaluate")


def dataset_hashes():
    a = pd.read_csv(ROOT / "reports/audit_files.csv", keep_default_na=False)
    a = a[a["ok"].astype(str) == "True"]
    return set(a.md5), np.array([int(h, 16) for h in a.phash_id], dtype=np.uint64), a.relpath.to_numpy()


def phash_dihedral(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        thumb = im.convert("RGB")
    thumb.thumbnail((256, 256), Image.Resampling.BILINEAR)
    return np.array([int(str(imagehash.phash(fn(thumb))), 16) for _, fn in audit.DIHEDRAL], dtype=np.uint64)


def grid(df, imgs, path, title, label_col=None, max_n=48):
    df = df.head(max_n)
    if df.empty:
        return
    cols = 6
    rows = int(np.ceil(len(df) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.1, rows * 2.5), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (i, r) in zip(axes.ravel(), df.iterrows()):
        ax.imshow(imgs[i])
        t = f"{Path(r.file).name[:20]}\n→ {ev.SHORT.get(r.pred, r.pred)} {r.conf:.2f}"
        if label_col and isinstance(r[label_col], str):
            t = f"[{ev.SHORT.get(r[label_col], r[label_col])}] " + t
        ax.set_title(t, fontsize=6, color="red" if (label_col and r[label_col] != r.pred) else "black")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", type=Path, default=ROOT / "data/uji_internet")
    ap.add_argument("--models", nargs="+", default=["mobilenetv3_large_100_augB", "yolo11n_cls"])
    ap.add_argument("--dup-threshold", type=int, default=4, help="jarak pHash <= ini = dianggap gambar dataset")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    folder = args.folder if args.folder.is_absolute() else ROOT / args.folder
    out = args.out or ROOT / "reports/uji_internet" / f"{dt.datetime.now():%Y%m%d_%H%M}"
    out.mkdir(parents=True, exist_ok=True)

    classes = load_classifier(args.models[0]).classes
    recs = []
    for sub in sorted(p for p in folder.iterdir() if p.is_dir()):
        kind = "label" if sub.name in classes else ("bukan_daun" if sub.name == "_bukan_daun" else "tanpa_label")
        if kind == "label" or sub.name in ("_belum_dilabeli", "_bukan_daun"):
            for f in sorted(p for p in sub.rglob("*") if p.suffix.lower() in IMG_EXTS):
                recs.append({"file": str(f.relative_to(folder)), "path": f, "grup": kind,
                             "label": sub.name if kind == "label" else None})
        elif not sub.name.startswith("."):
            print(f"Folder diabaikan (bukan nama kelas): {sub.name}")
    if not recs:
        raise SystemExit(f"Belum ada gambar di {folder}. Lihat PETUNJUK.txt")
    df = pd.DataFrame(recs)
    print(f"{len(df)} gambar: " + ", ".join(f"{k}={v}" for k, v in df.grup.value_counts().items()))

    # ---- baca gambar + cek kebocoran ----
    md5s, ds_hash, ds_path = dataset_hashes()
    imgs, ok, md5_hit, ddist, dnear = [], [], [], [], []
    for p in df.path:
        try:
            imgs.append(center_224(load_rgb(p, draft=None)))
            ok.append(True)
            md5_hit.append(hashlib.md5(p.read_bytes()).hexdigest() in md5s)
            h = phash_dihedral(p)
            d = np.bitwise_count(h[:, None] ^ ds_hash[None, :]).min(axis=0)
            j = int(d.argmin())
            ddist.append(int(d[j]))
            dnear.append(ds_path[j])
        except Exception as e:  # noqa: BLE001
            print(f"GAGAL dibaca: {p.name} ({type(e).__name__}: {e})")
            imgs.append(np.zeros((224, 224, 3), np.uint8))
            ok.append(False)
            md5_hit.append(False)
            ddist.append(64)
            dnear.append("")
    df = df.assign(terbaca=ok, md5_sama_dataset=md5_hit, jarak_phash_dataset=ddist, gambar_dataset_terdekat=dnear)
    df["dari_dataset"] = df.md5_sama_dataset | (df.jarak_phash_dataset <= args.dup_threshold)
    df = df[df.terbaca].reset_index(drop=True)
    imgs = [im for im, k in zip(imgs, ok) if k]

    L = [f"# Uji gambar eksternal — {dt.datetime.now():%Y-%m-%d %H:%M}", "",
         f"- Folder: `{folder.relative_to(ROOT).as_posix() if folder.is_relative_to(ROOT) else folder}`; "
         f"{len(df)} gambar terbaca ({', '.join(f'{k}={v}' for k, v in df.grup.value_counts().items())})",
         f"- **Terdeteksi berasal dari dataset Mendeley** (MD5 sama atau pHash ≤ {args.dup_threshold}, termasuk rotasi/flip): "
         f"**{int(df.dari_dataset.sum())}** gambar. Gambar ini sudah pernah 'dilihat' model (train/val/test), "
         "jadi metrik utama dihitung TANPA gambar tersebut.",
         "- Preprocessing sama dengan evaluasi: resize sisi terpendek 224 + center crop 224.", ""]
    if df.dari_dataset.any():
        L += ["| file | jarak pHash | gambar dataset terdekat |", "|---|---|---|"]
        L += [f"| {r.file} | {r.jarak_phash_dataset} | {r.gambar_dataset_terdekat} |"
              for r in df[df.dari_dataset].itertuples()] + [""]

    for name in args.models:
        clf = load_classifier(name)
        prob = ev.predict_all(clf, imgs, bs=32)
        d = df.assign(pred=[clf.classes[k] for k in prob.argmax(1)], conf=prob.max(1),
                      **{f"p_{c}": prob[:, i] for i, c in enumerate(clf.classes)})
        d.drop(columns=["path"]).to_csv(out / f"prediksi_{clf.name}.csv", index=False)
        L += [f"## Model: {clf.name}", ""]

        lab = d[d.grup == "label"]
        for judul, sub in (("TANPA gambar dari dataset (utama)", lab[~lab.dari_dataset]), ("semua gambar berlabel", lab)):
            if sub.empty:
                continue
            present = sorted(sub.label.unique())
            L += [f"### Berlabel — {judul}: n={len(sub)}", "",
                  f"- Akurasi: **{accuracy_score(sub.label, sub.pred):.3f}** ({int((sub.label == sub.pred).sum())}/{len(sub)} benar)",
                  f"- Macro-F1 (atas {len(present)} kelas yang ada di folder): "
                  f"**{f1_score(sub.label, sub.pred, labels=present, average='macro', zero_division=0):.3f}**", ""]
            rep = classification_report(sub.label, sub.pred, labels=present, output_dict=True, zero_division=0)
            L += [md_table(pd.DataFrame([{"kelas": c, "n": int(rep[c]["support"]), "recall": rep[c]["recall"],
                                          "precision": rep[c]["precision"], "f1": rep[c]["f1-score"]} for c in present]),
                           "{:.3f}"), ""]
        main_lab = lab[~lab.dari_dataset]
        if not main_lab.empty:
            y = main_lab.label.map({c: i for i, c in enumerate(clf.classes)}).to_numpy()
            p = main_lab.pred.map({c: i for i, c in enumerate(clf.classes)}).to_numpy()
            ev.plot_cm(y, p, clf.classes, out / f"cm_{clf.name}.png", f"{clf.name} — gambar eksternal (n={len(y)})")
            grid(main_lab.sort_values(["label", "file"]), imgs, out / f"grid_berlabel_{clf.name}.png",
                 f"{clf.name}: [label] → prediksi conf (merah = salah)", label_col="label")
            L += [f"Confusion matrix: `cm_{clf.name}.png`; grid gambar: `grid_berlabel_{clf.name}.png`", ""]

        for grup, judul in (("tanpa_label", "Belum dilabeli"), ("bukan_daun", "Bukan daun")):
            sub = d[d.grup == grup]
            if sub.empty:
                continue
            L += [f"### {judul}: n={len(sub)}", "",
                  f"- Confidence prediksi: median {sub.conf.median():.2f}, min {sub.conf.min():.2f}, maks {sub.conf.max():.2f}",
                  "- Prediksi: " + ", ".join(f"{ev.SHORT.get(k, k)}={v}" for k, v in sub.pred.value_counts().items()), ""]
            grid(sub, imgs, out / f"grid_{grup}_{clf.name}.png", f"{clf.name}: {judul} → prediksi conf")
        if (d.grup == "bukan_daun").any():
            L += ["Catatan: model hanya mengenal 6 kelas daun, sehingga gambar bukan daun tetap diberi salah satu label. "
                  "Confidence di atas berguna untuk menentukan ambang 'tidak yakin' (opsi A).", ""]

    (out / "ringkasan.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\nHasil tersimpan di: {out}")


if __name__ == "__main__":
    main()
