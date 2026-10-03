# -*- coding: utf-8 -*-
"""scripts/fusionner.py — lanceur CLI de la fusion par confiance.

Enveloppe mince de bebe.fusion (source unique) :
  python scripts/fusionner.py etats/local etats/etat_courant \\
      --heldout heldout.npz --out etats/fusionnee
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bebe.fusion import fusionner_confiance  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Fusion par confiance de paquets d'etat")
    ap.add_argument("etats", nargs="+", help="2 dossiers de paquets ou plus")
    ap.add_argument("--heldout", default=str(ROOT / "heldout.npz"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    r = fusionner_confiance(a.etats, a.heldout, a.out, a.device)
    print(json.dumps({k: v for k, v in r.items()
                      if k not in ("model", "mem", "hist")},
                     indent=2, ensure_ascii=False, default=str))
    sys.exit(0 if r.get("ok") else 1)


if __name__ == "__main__":
    main()
