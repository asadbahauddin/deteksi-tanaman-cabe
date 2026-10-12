"""Tahap 4: evaluasi akhir pada TEST set (sekali pakai).

Semua model menerima input identik: gambar asli -> resize sisi terpendek 224 (INTER_AREA) -> center crop 224.
Robustness: input yang sama diturunkan ke N px lalu dinaikkan lagi ke 224 (default 128, 96, 64).

Output (reports/eval/):
  metrics_<run>.json, classification_report_<run>.csv, cm_<run>.png (absolut + ternormalisasi),
  preds_<run>.csv, errors_<run>.png, per_style.csv, robustness.csv/.png, gradcam_<run>.png, gradcam_mass.csv,
  summary.md
Guard: menolak berjalan ulang pada split test bila reports/eval/TEST_USED ada (gunakan --force dengan alasan).
"""
import argparse
import datetime as dt
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score  # noqa: E402

from common import ROOT, SEED, md_table, seed_everything  # noqa: E402
from data_utils import center_224, load_rgb  # noqa: E402
from models_io import TimmClassifier, load_classifier  # noqa: E402

SHORT = {"Bacterial_Spot": "Bact", "Cercospora_Leaf_Spot": "Cerc", "Curl_Virus": "Curl",
         "Healthy_Leaf": "Healthy", "Nutrition_Deficiency": "Nutr", "Powdery_Mildew": "Powd"}


def degrade(img: np.ndarray, px: int) -> np.ndarray:
    small = cv2.resize(img, (px, px), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, img.shape[1::-1], interpolation=cv2.INTER_LINEAR)


def leaf_mask_white_bg(img: np.ndarray, min_bg: float = 0.15):
    """Mask daun = piksel bukan putih (min RGB <= 235), hanya untuk gambar berlatar putih.
    Kanal alpha PNG di dataset ini TIDAK transparan (semua alpha >= 128), jadi tidak bisa dipakai sebagai mask.
    Mengembalikan None bila latar putih < min_bg (foto berlatar abu/biru)."""
    white = img.min(axis=2) > 235
    if white.mean() < min_bg:
        return None
    k = np.ones((5, 5), np.uint8)
    leaf = cv2.morphologyEx((~white).astype(np.uint8), cv2.MORPH_OPEN, k)
    leaf = cv2.morphologyEx(leaf, cv2.MORPH_CLOSE, k)
    return leaf.astype(bool)


def predict_all(clf, imgs, bs=64):
    return np.concatenate([clf.predict_proba(imgs[i:i + bs]) for i in range(0, len(imgs), bs)])


def plot_cm(y, p, classes, path, title):
    cm = confusion_matrix(y, p, labels=range(len(classes)))
    cmn = cm / cm.sum(1, keepdims=True)
    labels = [SHORT.get(c, c) for c in classes]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axes[0], cbar=False)
    sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, xticklabels=labels, yticklabels=labels,
                ax=axes[1], cbar=False)
    for ax, t in zip(axes, ("absolut", "ternormalisasi per kelas asli")):
        ax.set_xlabel("prediksi")
        ax.set_ylabel("asli")
        ax.set_title(t)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return cm


def error_grid(df, imgs, classes, path, title, per_pair=6, max_pairs=8):
    err = df[df.label != df.pred]
    if err.empty:
        return []
    pairs = err.groupby(["label", "pred"]).size().sort_values(ascending=False)
    # pastikan Curl<->Nutrition tampil bila ada
    special = [p for p in pairs.index if {classes[p[0]], classes[p[1]]} == {"Curl_Virus", "Nutrition_Deficiency"}]
    order = special + [p for p in pairs.index if p not in special]
    order = order[:max_pairs]
    fig, axes = plt.subplots(len(order), per_pair, figsize=(per_pair * 1.9, len(order) * 2.2), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for r, (t, p) in enumerate(order):
        sub = err[(err.label == t) & (err.pred == p)].sort_values("conf", ascending=False).head(per_pair)
        for c, row in enumerate(sub.itertuples()):
            axes[r, c].imshow(imgs[row.Index])
            axes[r, c].set_title(f"{Path(row.path).name[:22]}\nconf {row.conf:.2f}", fontsize=6)
        axes[r, 0].text(-0.15, 0.5, f"{SHORT[classes[t]]}→{SHORT[classes[p]]}\n(n={pairs[(t, p)]})",
                        transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return [(classes[t], classes[p], int(pairs[(t, p)])) for t, p in pairs.index]


def gradcam_section(clf: TimmClassifier, df, imgs, classes, out_dir, rng, n_each=2):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    # layer target dipilih di VAL: EfficientNet -> bn2 (blocks[-1]/conv_head memberi 12-46% CAM kosong);
    # MobileNetV3 -> blocks[-1] (conv_head-nya setelah global pool, tidak spasial)
    layer = clf.model.bn2 if "efficientnet" in clf.cfg["model"] else clf.model.blocks[-1]
    cam = GradCAM(model=clf.model, target_layers=[layer])
    # 1) grid kualitatif: per kelas n_each benar + n_each salah (target = kelas prediksi)
    rows = []
    for ci, c in enumerate(classes):
        ok = df[(df.label == ci) & (df.pred == ci)]
        bad = df[(df.label == ci) & (df.pred != ci)]
        rows.append([("benar", i) for i in ok.sample(min(n_each, len(ok)), random_state=rng).index] +
                    [("salah", i) for i in bad.sample(min(n_each, len(bad)), random_state=rng).index])
    ncol = 2 * n_each
    fig, axes = plt.subplots(len(classes), ncol, figsize=(ncol * 2.1, len(classes) * 2.3), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for r, items in enumerate(rows):
        for c, (kind, i) in enumerate(items):
            g = cam(input_tensor=clf.to_tensor([imgs[i]]), targets=[ClassifierOutputTarget(int(df.pred[i]))])[0]
            axes[r, c].imshow(show_cam_on_image(imgs[i].astype(np.float32) / 255, g, use_rgb=True))
            axes[r, c].set_title(f"{kind}: {SHORT[classes[df.label[i]]]}→{SHORT[classes[df.pred[i]]]}", fontsize=7)
        axes[r, 0].text(-0.1, 0.5, SHORT[classes[r]], transform=axes[r, 0].transAxes, ha="right", va="center")
    fig.suptitle(f"Grad-CAM {clf.name} (kiri: benar, kanan: salah; target = kelas prediksi)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / f"gradcam_{clf.name}.png", dpi=100)
    plt.close(fig)

    # 2) kuantitatif: fraksi massa CAM di dalam daun (mask alpha PNG) vs fraksi area daun
    recs = []
    idx = df.index[df.mask_ok]  # subset berlatar putih
    for k in range(0, len(idx), 32):
        b = list(idx[k:k + 32])
        gs = cam(input_tensor=clf.to_tensor([imgs[i] for i in b]),
                 targets=[ClassifierOutputTarget(int(df.pred[i])) for i in b])
        for i, g in zip(b, gs):
            m = df.at[i, "mask"]
            tot = g.sum()
            recs.append({"run": clf.name, "path": df.path[i], "label": classes[df.label[i]],
                         "correct": bool(df.label[i] == df.pred[i]), "leaf_area": float(m.mean()),
                         "cam_empty": bool(tot < 1e-6),
                         "cam_mass_on_leaf": float(g[m].sum() / tot) if tot >= 1e-6 else np.nan,
                         "cam_mass_corner_tl": float(g[:32, :32].sum() / tot) if tot >= 1e-6 else np.nan})
    return pd.DataFrame(recs)


def run_gradcam_only(args, out):
    df = pd.read_csv(ROOT / f"splits/{args.split}.csv")
    imgs = [center_224(load_rgb(ROOT / p, draft=None)) for p in df.path]
    masks = [leaf_mask_white_bg(im) for im in imgs]
    frames = []
    for r in args.runs:
        clf = load_classifier(r)
        if not isinstance(clf, TimmClassifier):
            continue
        classes = clf.classes
        y = df.label.map({c: i for i, c in enumerate(classes)}).to_numpy()
        prob = predict_all(clf, imgs)
        d = df.assign(label=y, pred=prob.argmax(1), conf=prob.max(1), mask=masks)
        d["mask_ok"] = d["mask"].map(lambda m: m is not None)
        frames.append(gradcam_section(clf, d, imgs, classes, out, np.random.RandomState(SEED)))
    C = pd.concat(frames)
    C.to_csv(out / "gradcam_mass.csv", index=False)
    cg = C.groupby(["run", "correct"]).agg(n=("path", "size"), cam_kosong=("cam_empty", "sum"),
                                            leaf_area=("leaf_area", "mean"), cam_on_leaf=("cam_mass_on_leaf", "mean"),
                                            cam_on_leaf_median=("cam_mass_on_leaf", "median"),
                                            cam_pojok_kiri_atas=("cam_mass_corner_tl", "mean")).reset_index()
    txt = md_table(cg, "{:.3f}")
    (out / "gradcam_summary.md").write_text(
        "# Grad-CAM (layer: EfficientNet=bn2, MobileNetV3=blocks[-1])\n\n" + txt + "\n", encoding="utf-8")
    print(txt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="nama folder di runs/ (mis. yolo11n_cls efficientnet_b0_augB)")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--degrade-px", nargs="+", type=int, default=[128, 96, 64])
    ap.add_argument("--out", type=Path, default=ROOT / "reports/eval")
    ap.add_argument("--force", type=str, default="", help="alasan menjalankan ulang pada test (dicatat)")
    ap.add_argument("--gradcam-only", action="store_true",
                    help="hanya Grad-CAM (tanpa metrik/robustness); untuk koreksi layer CAM, dicatat di TEST_USED")
    args = ap.parse_args()
    seed_everything(SEED)
    out = args.out if args.split == "test" else args.out.with_name(args.out.name + "_val")
    out.mkdir(parents=True, exist_ok=True)
    flag = out / "TEST_USED"
    if args.gradcam_only:
        if args.split == "test":
            with flag.open("a", encoding="utf-8") as f:
                f.write(f"{dt.datetime.now():%Y-%m-%d %H:%M} GRADCAM-ONLY runs={args.runs} "
                        "(koreksi layer target CAM, dipilih di val; metrik tidak dihitung ulang)\n")
        run_gradcam_only(args, out)
        return
    if args.split == "test" and flag.exists() and not args.force:
        raise SystemExit(f"Test set sudah dipakai ({flag.read_text(encoding='utf-8').strip()}). "
                         "Gunakan --force \"<alasan>\" bila benar-benar perlu.")
    if args.split == "test":
        with flag.open("a", encoding="utf-8") as f:
            f.write(f"{dt.datetime.now():%Y-%m-%d %H:%M} runs={args.runs} force={args.force!r}\n")

    df = pd.read_csv(ROOT / f"splits/{args.split}.csv")
    print(f"Memuat {len(df)} gambar {args.split} ...", flush=True)
    imgs = [center_224(load_rgb(ROOT / p, draft=None)) for p in df.path]
    masks = [leaf_mask_white_bg(im) for im in imgs]
    clfs = [load_classifier(r) for r in args.runs]
    classes = clfs[0].classes
    assert all(c.classes == classes for c in clfs), "urutan kelas antar model berbeda"
    y = df.label.map({c: i for i, c in enumerate(classes)}).to_numpy()

    summary, per_style, robust, cam_frames, err_pairs = [], [], [], [], {}
    for clf in clfs:
        print(f"== {clf.name}", flush=True)
        prob = predict_all(clf, imgs)
        pred = prob.argmax(1)
        d = df.assign(label=y, pred=pred, conf=prob.max(1), mask=masks)
        d["mask_ok"] = d["mask"].map(lambda m: m is not None)
        d.drop(columns=["mask"]).assign(**{f"p_{c}": prob[:, i] for i, c in enumerate(classes)}) \
            .to_csv(out / f"preds_{clf.name}.csv", index=False)
        rep = classification_report(y, pred, target_names=classes, digits=4, output_dict=True)
        pd.DataFrame(rep).T.to_csv(out / f"classification_report_{clf.name}.csv")
        cm = plot_cm(y, pred, classes, out / f"cm_{clf.name}.png", f"{clf.name} — {args.split} (n={len(y)})")
        m = {"run": clf.name, "kind": clf.kind, "n": len(y), "accuracy": accuracy_score(y, pred),
             "macro_f1": f1_score(y, pred, average="macro"),
             "per_class": {c: {k: rep[c][k] for k in ("precision", "recall", "f1-score", "support")} for c in classes},
             "confusion_matrix": cm.tolist()}
        (out / f"metrics_{clf.name}.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
        summary.append({"run": clf.name, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                        **{f"F1_{SHORT[c]}": rep[c]["f1-score"] for c in classes}})
        err_pairs[clf.name] = error_grid(d, imgs, classes, out / f"errors_{clf.name}.png",
                                         f"{clf.name}: salah klasifikasi per pasangan (asli→prediksi)")
        for s, g in d.groupby("style"):
            per_style.append({"run": clf.name, "style": s, "n": len(g), "salah": int((g.label != g.pred).sum()),
                              "accuracy": accuracy_score(g.label, g.pred),
                              "kelas": ",".join(sorted({SHORT[classes[k]] for k in g.label}))})
        robust.append({"run": clf.name, "px": "asli(224)", "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]})
        for px in args.degrade_px:
            pp = predict_all(clf, [degrade(im, px) for im in imgs]).argmax(1)
            robust.append({"run": clf.name, "px": str(px), "accuracy": accuracy_score(y, pp),
                           "macro_f1": f1_score(y, pp, average="macro")})
        if isinstance(clf, TimmClassifier):
            cam_frames.append(gradcam_section(clf, d, imgs, classes, out, np.random.RandomState(SEED)))

    S = pd.DataFrame(summary)
    S.to_csv(out / "summary.csv", index=False)
    P = pd.DataFrame(per_style)
    P.to_csv(out / "per_style.csv", index=False)
    R = pd.DataFrame(robust)
    R.to_csv(out / "robustness.csv", index=False)
    Rw = R.pivot(index="run", columns="px", values="macro_f1")[["asli(224)"] + [str(p) for p in args.degrade_px]]
    fig, ax = plt.subplots(figsize=(7, 4))
    for run, row in Rw.iterrows():
        ax.plot(row.index, row.values, marker="o", label=run)
    ax.set_xlabel("resolusi efektif (px, lalu di-upscale ke 224)")
    ax.set_ylabel(f"macro-F1 ({args.split})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "robustness.png", dpi=110)
    plt.close(fig)

    L = [f"# Evaluasi {args.split} set", "", f"- n = {len(df)}; waktu {dt.datetime.now():%Y-%m-%d %H:%M}; runs: {', '.join(args.runs)}",
         "- Input identik untuk semua model: resize sisi terpendek 224 (INTER_AREA) + center crop 224.", "",
         "## Ringkasan", "", md_table(S, "{:.4f}"), "",
         "## Robustness (macro-F1)", "", md_table(Rw.reset_index(), "{:.4f}"), "",
         "## Per gaya/sub-sumber", "", md_table(P, "{:.4f}"), "", "## Pasangan salah klasifikasi", ""]
    for run, prs in err_pairs.items():
        L.append(f"**{run}**: " + ("; ".join(f"{SHORT[t]}→{SHORT[p]} {n}" for t, p, n in prs) or "tidak ada"))
        L.append("")
    if cam_frames:
        C = pd.concat(cam_frames)
        C.to_csv(out / "gradcam_mass.csv", index=False)
        cg = C.groupby(["run", "correct"]).agg(n=("path", "size"), cam_kosong=("cam_empty", "sum"),
                                                leaf_area=("leaf_area", "mean"),
                                                cam_on_leaf=("cam_mass_on_leaf", "mean"),
                                                cam_on_leaf_median=("cam_mass_on_leaf", "median"),
                                                cam_pojok_kiri_atas=("cam_mass_corner_tl", "mean")).reset_index()
        L += ["## Grad-CAM: fraksi massa CAM di dalam daun (subset berlatar putih, mask = piksel bukan putih)", "",
              "Bila model fokus pada daun, `cam_on_leaf` jauh di atas `leaf_area` (baseline acak = luas daun). "
              "`cam_pojok_kiri_atas` = massa CAM di blok 32x32 kiri-atas (2% area; baseline acak ~0.02).", "",
              md_table(cg, "{:.3f}"), ""]
    (out / "summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
