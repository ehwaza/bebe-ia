# -*- coding: utf-8 -*-
"""bebe — package du bebe-ia : modele, metriques, fusion par confiance.

SOURCE UNIQUE : les classes vivent dans banc.py (l'instrument de mesure).
Ce package les RE-EXPORTE pour ne jamais dupliquer le code du genome.
"""
from bebe.modele import (  # noqa: F401
    ALPHABET, VOCAB, NV, CAP_MEMOIRE, DEVICE,
    Bebe, MemoireKNN, MemoireConsolidee,
    gen_item, to_ids, creer_bebe,
)
from bebe.metriques import ece_score, risk_coverage, resume_metriques  # noqa: F401
