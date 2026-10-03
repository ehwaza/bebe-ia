# -*- coding: utf-8 -*-
"""bebe/modele.py — le genome du bebe : classes et symboles.

POLITIQUE DE NON-DUPLICATION (Mathieu, 04/10) : banc.py est l'instrument
de mesure INDEPENDANT et la SOURCE UNIQUE des classes. On re-exporte ici,
on ne recopie pas. Aucune divergence possible entre le banc et l'entraineur.

Classes :
  Bebe              transformer minimal (2 couches, d=128), sans pre-entrainement
  MemoireKNN        memoire externe simple (append-only)
  MemoireConsolidee memoire consolidee PAR CLE (format bebe-etat/1)
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import banc  # noqa: E402  (source unique du genome)

ALPHABET = banc.ALPHABET
VOCAB = banc.VOCAB
NV = banc.NV
CAP_MEMOIRE = banc.CAP_MEMOIRE
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

Bebe = banc.Bebe
MemoireKNN = banc.MemoireKNN
MemoireConsolidee = banc.MemoireConsolidee
gen_item = banc.gen_item
to_ids = banc.to_ids


def creer_bebe(seed_init, arch=None, device="cpu"):
    """Bebe instancie avec une graine d'init explicite (reproductibilite)."""
    import torch
    torch.manual_seed(int(seed_init))
    arch = arch or {"d": 128, "nl": 2, "heads": 4}
    return banc.Bebe(d=int(arch["d"]), nl=int(arch["nl"]),
                     heads=int(arch["heads"])).to(device)
