# -*- coding: utf-8 -*-
"""scripts/multi_seeds.py — 5 runs a graines differentes + agregation.

Chaque run s'ecrit dans resultats/seed_<n>/ (seed, config, resultats tracés).
Agregation : moyenne + variance -> courbes_multi_seeds.png + resultats_multi_seeds.md.

Usage : python scripts/multi_seeds.py [--runs 5] [--n 50000] [--seeds 1234,1235,...]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def lancer_run(seed, n, out, source, regles=None, central_spec=None):
    cmd = [sys.executable, str(ROOT / "scripts" / "entrainer_local.py"),
           "--n", str(n), "--seed", str(seed), "--seed-data", str(seed * 7919 + 13),
           "--out", str(out), "--source", source, "--eval-every", "2000"]
    if regles:                      # shard de regles (defaut None = toutes = neutre)
        cmd += ["--regles", str(regles)]
    if central_spec:                # controle central concat pools (defaut None = neutre)
        cmd += ["--central-spec", str(central_spec)]
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--seeds", default=None, help="ex: 1234,1235,1236,1237,1238")
    ap.add_argument("--source", default="synthetique")
    ap.add_argument("--regles", default=None,
                    help="shard de regles (ex: 'rep') = meme pass-through que fragment")
    ap.add_argument("--central-spec", default=None,
                    help="JSON central-spec (concat pools) = meme pass-through")
    ap.add_argument("--out-root", default=str(ROOT / "resultats"))
    a = ap.parse_args()

    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds \
        else [1234 + i for i in range(a.runs)]
    out_root = Path(a.out_root)

    # --- runs (chaque run dans seed_<n>/, seed+config traces dedans)
    for s in seeds:
        d = out_root / f"seed_{s}"
        if (d / "resultats.json").exists():
            print(f"seed_{s} deja present : saute (supprime le dossier pour relancer)")
            continue
        lancer_run(s, a.n, d, a.source, regles=a.regles,
                   central_spec=a.central_spec)

    # --- agregation : moyenne + variance
    traces, resumes = [], []
    for s in seeds:
        d = out_root / f"seed_{s}"
        r = json.loads((d / "resultats.json").read_text(encoding="utf-8"))
        resumes.append(r["resume"])
        z = np.load(d / "hist.npz")
        traces.append(np.asarray(z["ok"], float))
    L = min(len(t) for t in traces)
    M = np.stack([t[:L] for t in traces])              # (runs, L)
    moy_cum = np.cumsum(M, axis=1) / np.arange(1, L + 1)
    mean, std = moy_cum.mean(0), moy_cum.std(0)

    # --- courbe multi-seeds
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = np.arange(1, L + 1)
    plt.figure(figsize=(9, 5))
    plt.plot(xs, mean, color="tab:blue", label="accuracy cumulee (moyenne)")
    plt.fill_between(xs, mean - std, mean + std, color="tab:blue",
                     alpha=0.25, label="± 1 ecart-type")
    plt.xlabel("interactions")
    plt.ylabel("accuracy cumulee")
    plt.title(f"Multi-seeds — {len(seeds)} runs x {a.n} interactions")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_root / "courbes_multi_seeds.png", dpi=120)
    plt.close()

    # --- rapport moyenne + variance
    cles = ["ece", "accuracy", "conf_moy", "aurc", "coverage_0p5", "ms_per_iter"]
    lignes = [f"# Multi-seeds — {len(seeds)} runs", "",
              f"Seeds : {seeds} | n = {a.n} | source = {a.source}", "",
              "| metrique | moyenne | ecart-type |", "|---|---|---|"]
    for c in cles:
        vals = np.array([r.get(c, float("nan")) for r in resumes], float)
        lignes.append(f"| {c} | {np.nanmean(vals):.4f} | {np.nanstd(vals):.4f} |")
    lignes += ["", "## Par run", "",
               "| seed | ECE | accuracy | AURC | ms/iter |", "|---|---|---|---|---|"]
    for s, r in zip(seeds, resumes):
        lignes.append(f"| {s} | {r.get('ece', float('nan')):.4f} "
                      f"| {r.get('accuracy', float('nan')):.4f} "
                      f"| {r.get('aurc', float('nan')):.4f} "
                      f"| {r.get('ms_per_iter', float('nan')):.2f} |")
    (out_root / "resultats_multi_seeds.md").write_text(
        "\n".join(lignes), encoding="utf-8")
    print("\n".join(lignes))
    print("\nsorties :", out_root / "courbes_multi_seeds.png",
          "|", out_root / "resultats_multi_seeds.md")


if __name__ == "__main__":
    main()
