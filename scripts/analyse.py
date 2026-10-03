# -*- coding: utf-8 -*-
"""scripts/analyse.py — synthese finale de TOUTES les runs.

Charge tous les resultats (resultats/*/resultats.json + etats/*/run_github.json),
calcule moyennes et ecarts-types de : ECE final, accuracy, AURC, coverage.
Ecrit resultats_final.md.

Usage : python scripts/analyse.py [--racine resultats]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def collecter(racine):
    """Tous les resumes disponibles, tracables jusqu'a leur fichier source."""
    entrees = []
    for f in sorted(Path(racine).rglob("resultats.json")):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))["resume"]
            entrees.append((str(f.relative_to(racine)), r))
        except Exception:
            pass
    for f in sorted(Path(ROOT / "etats").rglob("run_github.json")):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
            entrees.append((str(f.relative_to(ROOT)), {
                "ece": r.get("ece_heldout", float("nan")),
                "ms_per_iter": r.get("ms_per_iter"),
                "accuracy": float("nan"), "aurc": float("nan"),
                "coverage_0p5": float("nan"), "n": r.get("n_new", 0)}))
        except Exception:
            pass
    return entrees


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--racine", default=str(ROOT / "resultats"))
    ap.add_argument("--out", default=str(ROOT / "resultats_final.md"))
    a = ap.parse_args()

    entrees = collecter(a.racine)
    if not entrees:
        print("aucun resultat trouve sous", a.racine)
        sys.exit(1)

    cles = ["ece", "accuracy", "aurc", "coverage_0p5", "conf_moy",
            "ms_per_iter", "n"]
    lignes = ["# Resultats finaux — bebe-ia", "",
              f"Runs analysees : {len(entrees)}", "",
              "| metrique | moyenne | ecart-type | min | max | n |",
              "|---|---|---|---|---|---|"]
    for c in cles:
        vals = np.array([r.get(c, float("nan")) for _, r in entrees], float)
        vals = vals[np.isfinite(vals)] if c != "n" else vals
        if len(vals) == 0:
            continue
        lignes.append(f"| {c} | {np.mean(vals):.4f} | {np.std(vals):.4f} "
                      f"| {np.min(vals):.4f} | {np.max(vals):.4f} | {len(vals)} |")

    lignes += ["", "## Detail (tracabilite : seed, config, resultats)", "",
               "| source | ECE | accuracy | AURC | coverage@0.5 | ms/iter | n |",
               "|---|---|---|---|---|---|---|"]
    for src, r in entrees:
        lignes.append(
            f"| {src} | {r.get('ece', float('nan')):.4f} "
            f"| {r.get('accuracy', float('nan')):.4f} "
            f"| {r.get('aurc', float('nan')):.4f} "
            f"| {r.get('coverage_0p5', float('nan')):.4f} "
            f"| {r.get('ms_per_iter', float('nan')):.2f} "
            f"| {r.get('n', 0)} |")

    lignes += ["", "Lecture : ECE bas = le bebe SAIT quand il ne sait pas ;",
               "AURC bas + coverage haut = il repond honnetement sur le plus grand",
               "nombre d'items possibles. Les deux ensemble = calibration en emergence."]
    Path(a.out).write_text("\n".join(lignes), encoding="utf-8")
    print("\n".join(lignes))
    print("\necrit :", a.out)


if __name__ == "__main__":
    main()
