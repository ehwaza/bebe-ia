# -*- coding: utf-8 -*-
"""Test du writer de nectar : ecriture, roundtrip, refus (alteration/genome)."""
import json
from types import SimpleNamespace

import numpy as np

from bebe import nectar

rng = np.random.default_rng(0)
m, k, c = 40, 64, 5

yc = rng.integers(0, 9, (m, c)).astype(np.int32)
mem = SimpleNamespace(
    keys=rng.standard_normal((m, k)).astype(np.float32),
    y_counts=yc,
    n=yc.sum(axis=1).astype(np.int32),
    ok=rng.random(m).astype(np.float32),
    last_seen=np.arange(m, dtype=np.int64) * 7,
)

meta = {"heldout_sha256": "a" * 64, "code_sha": "b" * 64,
        "arch": {"d_model": 64, "layers": 2},
        "parent": "root", "seeds": {"data": 7}}  # parent/seed: hors genome

ligne = nectar.ligne_nectar(
    abeille="ab01deadbeef0001",
    niche="regles=rep|arith; data_seed=7",
    meta=meta, seen={"n_items": 1234, "cycles": 42}, mem=mem, m_total=m + 12)

ok, raisons = nectar.valider(ligne)
assert ok, f"ligne valide refusee: {raisons}"
print(f"[1] ligne valide : OK  (m={ligne['nectar']['m']}, "
      f"m_total={ligne['nectar']['m_total']}, h={ligne['h'][:16]}...)")

# --- roundtrip : les 5 tableaux survivent identiques + ok = ok_num/n
keys, yc2, n2, ok_num, ls = nectar.lire_nectar(ligne)
assert np.allclose(keys, mem.keys, atol=0), "keys alteres"
assert np.array_equal(yc2, mem.y_counts), "y_counts alteres"
assert np.array_equal(n2, mem.n), "n alteres"
assert np.allclose(ok_num / n2, mem.ok, atol=1e-6), "ok=ok_num/n casse"
assert np.array_equal(ls, mem.last_seen), "last_seen alteres"
print("[2] roundtrip : keys/y_counts/n/ok_num/last_seen identiques, "
      "ok = ok_num/n reconstruit  OK")

# --- altération d'octet -> hash invalide
import base64
l2 = json.loads(json.dumps(ligne))
tampered = bytearray(base64.b64decode(l2["nectar"]["n"]))
tampered[0] ^= 0xFF
l2["nectar"]["n"] = base64.b64encode(bytes(tampered)).decode()
ok2, r2 = nectar.valider(l2)
assert not ok2 and any("hash" in x for x in r2), f"alteration non detectee: {r2}"
print(f"[3] alteration detectee : {r2}")

# --- genome incomplet -> refuse
l3 = json.loads(json.dumps(ligne))
del l3["genome"]["code_sha"]
ok3, r3 = nectar.valider(l3)
assert not ok3 and any("genome" in x for x in r3), f"genome non gate: {r3}"
print(f"[3b] genome gate : {r3}")

# --- niche mal formee -> refuse
l4 = json.loads(json.dumps(ligne))
l4["niche"] = "tout le monde"
l4["h"] = nectar._h(l4)  # re-hasher ne sauve pas la niche
ok4, r4 = nectar.valider(l4)
assert not ok4 and any("niche" in x for x in r4), f"niche non gatee: {r4}"
print(f"[3c] niche gatee : {r4}")

# --- ok_num > n (compteur byzantin) + re-hash -> refuse quand meme
l5 = json.loads(json.dumps(ligne))
l5["nectar"]["ok_num"] = base64.b64encode(
    (mem.n.astype(np.float32) * 3).tobytes()).decode()
l5["h"] = nectar._h(l5)  # l'attaquant re-hache : la coherence comptable rattrape
ok5, r5 = nectar.valider(l5)
assert not ok5 and any("ok_num" in x for x in r5), f"ok_num byzantin accepte: {r5}"
print(f"[3d] ok_num byzantin : {r5}")

# --- abeille hors format hex16 -> refuse
l6 = json.loads(json.dumps(ligne))
l6["abeille"] = "AbeilleNaise"
l6["h"] = nectar._h(l6)
ok6, r6 = nectar.valider(l6)
assert not ok6 and any("hex16" in x for x in r6), f"abeille non gatee: {r6}"
print(f"[3e] abeille hex16 gatee : {r6}")

# --- jsonl : ecrit + relit
import tempfile, os
p = os.path.join(tempfile.mkdtemp(), "miel.jsonl")
nectar.ecrire_jsonl(p, [ligne, ligne])
back = nectar.lire_jsonl(p)
assert len(back) == 2 and back[0]["h"] == ligne["h"], "jsonl casse"
print("[4] jsonl append + lecture : OK")

print("\nTOUS LES TESTS PASSENT — writer pret pour le premier fold de bout en bout")
