# -*- coding: utf-8 -*-
"""Tests LOGIQUE PURE de la cible miel Phase 2 (PROTOCOLE_PHASE2.md s3).

AUCUN entraînement : formule de cible, equivalences de loss, chargement et
consult du miel fixe, telemetrie P6. Verif de code, pas un run.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

R = Path(r"F:\becvide\colab_ready\bebe-ia")
sys.path.insert(0, str(R / "scripts"))
sys.path.insert(0, str(R))

from entrainer_local import cible_miel, _charger_miel, _resume_miel  # noqa: E402

NV = 39

# --- [1] sk <= 0.5 -> AUCUNE modulation (None) ------------------------------
assert cible_miel(NV, 3, 7, 0.5) is None, "seuil strict : sk==0.5 -> None"
assert cible_miel(NV, 3, 7, 0.4999) is None, "sk<0.5 -> None"
assert cible_miel(NV, 3, None, 0.9) is None, "maj=None -> None"
print("[1] seuil 0.5 strict (spec DUR) + maj=None : OK")

# --- [2] sk > 0.5 -> vecteur de probabilites, masse exacte -------------------
t = cible_miel(NV, 3, 7, 0.8)
assert t is not None and t.dtype == np.float32 and t.shape == (NV,)
assert abs(float(t.sum()) - 1.0) < 1e-6, f"somme != 1 : {t.sum()}"
assert abs(float(t[7]) - 0.8) < 1e-6 and abs(float(t[3]) - 0.2) < 1e-6
assert float(t.sum() - t[7] - t[3]) == 0.0, "autres classes doivent etre 0"
print("[2] sk*onehot(maj)+(1-sk)*onehot(y) : OK (masse, dtype, support)")

# --- [3] cas maj == y : la masse se recombine a 1.0 -------------------------
t = cible_miel(NV, 5, 5, 0.7)
assert abs(float(t[5]) - 1.0) < 1e-6 and abs(float(t.sum()) - 1.0) < 1e-6
print("[3] maj==y -> onehot(y) recombine : OK")

# --- [4] bornes de sk -------------------------------------------------------
t = cible_miel(NV, 1, 2, 0.5001)
assert float(t[1]) >= 0.4999 - 1e-6 and float(t[2]) <= 0.5001 + 1e-6
t = cible_miel(NV, 1, 2, 0.999)
assert float(t[2]) <= 1.0 + 1e-6 and float(t[1]) >= -1e-6
print("[4] bornes de sk : OK")

# --- [5] equivalence de loss : onehot vs index de classe --------------------
g = torch.Generator().manual_seed(0)
logits = torch.randn(1, NV, dtype=torch.float32, generator=g)
y = 4
l_index = torch.nn.CrossEntropyLoss()(logits, torch.tensor([y]))
onehot = torch.zeros(1, NV)
onehot[0, y] = 1.0
l_soft = torch.nn.CrossEntropyLoss()(logits, onehot)
assert torch.allclose(l_index, l_soft, atol=1e-6), (l_index, l_soft)
t = cible_miel(NV, y, 9, 0.6)
l_mod = torch.nn.CrossEntropyLoss()(logits, torch.from_numpy(t).unsqueeze(0))
assert torch.isfinite(l_mod) and float(l_mod) > 0.0
print(f"[5] loss index==onehot ({float(l_index):.6f}), loss modulee finie "
      f"({float(l_mod):.6f}) : OK")

# --- [6] chargement miel fixe + consult RELLE, format npz du contrat --------
import banc  # noqa: E402

srcs = [c * 4 for c in "abcdefgh"]              # 8 srcs -> 8 vecs orthogonaux
keys = np.stack([np.asarray(banc.MemoireConsolidee.vec(s), np.float32)
                 for s in srcs])
m0 = len(srcs)
assert len({k.tobytes() for k in keys}) == m0, "VOCAB doit contenir a-h"
yc = np.zeros((m0, NV), np.int32)
for i in range(1, m0):
    yc[i, i] = 1
yc[0, 31] = 100                                 # cle 0 domine la classe 31
n = np.full((m0,), 10, np.int32)
ok = np.full((m0,), 0.9, np.float32)
p = Path(tempfile.mkdtemp()) / "miel_test.npz"
np.savez(p, keys=keys, y_counts=yc, n=n, ok=ok)
mem, mm = _charger_miel(p)
assert mm == m0
assert mem.keys.dtype == np.float32, "cast f32 = contrat train/eval identiques"
assert mem.keys.shape == (m0, NV)
maj, sk = mem.consult("aaaa")                   # hit exact sur la cle 0
assert maj == 31, f"vote attendu 31, obtenu {maj}"
assert sk > 0.5, f"sk>0.5 attendu (fiab .9 x accord ~.96), obtenu {sk}"
maj2, sk2 = mem.consult("zzzz")                 # pas de hit : sk incertain
assert maj2 is not None and 0.0 <= sk2 <= 1.0
print(f"[6] chargement npz + consult RELLE (maj=31, sk={sk:.3f}) : OK")

# --- [7] telemetrie P6 ------------------------------------------------------
r = _resume_miel(p, "sha", mm, 100, 12, [0.1] * 88 + [0.7] * 12, "code")
assert r["consultes"] == 100 and r["actifs"] == 12
assert abs(r["taux_activation"] - 0.12) < 1e-12
assert r["sk_p99"] >= r["sk_p90"] >= r["sk_p50"]
assert r["code_sha_entre"] == "code" and r["m"] == m0
r0 = _resume_miel(p, "sha", mm, 0, 0, [], "code")
assert r0["taux_activation"] != r0["taux_activation"], "NaN attendu si 0 consult"
print("[7] telemetrie P6 (taux, fractiles, NaN si vide) : OK")

print("\nTOUS LES TESTS PASSENT — cible miel conforme au protocole s3")
