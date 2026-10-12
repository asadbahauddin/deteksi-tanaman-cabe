"""Append satu baris eksperimen ke notes/eksperimen.md."""
import datetime as dt

from common import ROOT

HEADER = ("# Log eksperimen\n\n| waktu | model | augmentasi | config | best epoch | waktu train (menit) "
          "| val macro-F1 | val acc | run |\n|---|---|---|---|---|---|---|---|---|\n")


def append_experiment(res: dict, cfg_desc: str) -> None:
    path = ROOT / "notes/eksperimen.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(HEADER, encoding="utf-8")
    row = (f"| {dt.datetime.now():%Y-%m-%d %H:%M} | {res['model']} | {res['aug']} | {cfg_desc} | {res['best_epoch']} "
           f"| {res['train_min']} | {res['val_macro_f1']:.4f} | {res['val_acc']:.4f} | `{res['run_dir']}` |\n")
    with path.open("a", encoding="utf-8") as f:
        f.write(row)
