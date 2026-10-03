# -*- coding: utf-8 -*-
"""scripts/entrainer_github.py — le FRAGMENT GITHUB qui apprend en parallele.

Cable pour .github/workflows/apprendre.yml :
  1. Reprend le dernier etat depuis etats/etat_courant (le depot EST la memoire
     du fragment : chaque cycle reprend ou le precedent s'est arrete).
  2. Entraine pendant le BUDGET TEMPS (300 min, cap 6h GitHub). Le nombre
     d'interactions est un PLAFOND haut, pas le reglage : c'est le temps qui
     coupe. Zero calcul de debit, zero gaspillage de budget.
  3. Mesure le debit REEL (ms/iter) et l'ecrit dans meta.metrics :
     le run suivant (et LYNX) lisent la valeur exacte du runner.
  4. Sauvegarde l'etat mis a jour dans etats/etat_courant (commit par le workflow).

Estimation (commentaire, pas un reglage) :
  n_estime ~= floor(budget_s x 0.9 / ms_per_iter). Ex: 5 h a 4.6 ms/it ~= 3.9 M ;
  sur runner 2x plus lent ~1.9 M. Le budget s'adapte tout seul.

Usage : python scripts/entrainer_github.py [--budget-seconds 18000] [--plafond 5000000]
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from entrainer_local import boucle_entrainement  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-seconds", type=float, default=18000,
                    help="300 min par defaut, cap dur 6h GitHub")
    ap.add_argument("--plafond", type=int, default=5_000_000,
                    help="plafond d'interactions (le temps coupe avant)")
    ap.add_argument("--out", default=str(ROOT / "etats" / "etat_courant"))
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--seed-data", type=int, default=None,
                    help="unique par fragment ; defaut = derive (trace dans meta)")
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--ckpt-every", type=int, default=50000)
    a = ap.parse_args()

    out = Path(a.out)
    parent = out / "etat" if (out / "etat" / "meta.json").exists() else None
    if parent is None and (out / "meta.json").exists():
        parent = out  # compat : etat directement dans etats/etat_courant

    t0 = time.time()
    r = boucle_entrainement(
        n=a.plafond, out_dir=out, seed=a.seed,
        seed_data=a.seed_data or (int(time.time()) & 0x7FFFFFFF),
        source="synthetique", heldout_path=ROOT / "heldout.npz",
        budget_s=a.budget_seconds, eval_every=a.eval_every,
        ckpt_every=a.ckpt_every, parent_dir=parent, verbose=True)
    duree = time.time() - t0

    # debit reel du RUNNER -> trace pour caler les runs suivants
    rapport = {"duree_s": duree, "ms_per_iter": r.get("ms_per_iter"),
               "items_per_s": r.get("items_per_s"), "n_new": r.get("n_new"),
               "cycle": r.get("cycle"), "ece_heldout": r.get("ece")}
    (out / "run_github.json").write_text(
        json.dumps(rapport, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rapport, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
