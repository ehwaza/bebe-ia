# -*- coding: utf-8 -*-
"""bebe/metriques.py — mesure de calibration et de couverture.

SOURCE UNIQUE : banc.py. Re-export + synthese pour analyse.py / multi_seeds.py.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import banc  # noqa: E402

ece_score = banc.ece_score
risk_coverage = banc.risk_coverage


def resume_metriques(confs, oks):
    """Synthese complete d'une trace (confiance, exactitude).

    Retourne : ece, accuracy, conf_moy, aurc, coverage@0.5, n.
    Utilisee par analyse.py et multi_seeds.py pour les rapports."""
    confs = np.asarray(confs, float)
    oks = np.asarray(oks, float)
    if len(confs) == 0:
        return {"ece": float("nan"), "accuracy": float("nan"),
                "conf_moy": float("nan"), "aurc": float("nan"),
                "coverage_0p5": float("nan"), "n": 0}
    taus, cov, risk, aurc = banc.risk_coverage(confs, oks)
    return {
        "ece": float(banc.ece_score(confs, oks)),
        "accuracy": float(oks.mean()),
        "conf_moy": float(confs.mean()),
        "aurc": float(aurc),
        "coverage_0p5": float((confs >= 0.5).mean()),
        "n": int(len(confs)),
    }
