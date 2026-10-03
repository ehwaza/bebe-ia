# -*- coding: utf-8 -*-
"""etat.py — FORMAT D'ETAT du bebe : un paquet de FRAGMENT. (v1.1)

v1.1 integre le retour emetteur de LYNX (23:30) :
  - memoire CONSOLIDEE PAR CLE (une cle = un contexte, compteurs) -- format FIGE v1
  - invariants durs ajoutes : code_sha, arch, hist-local (pas de double compte),
    seed_data unique par (fragment, cycle)

Paquet :
  model.pt      poids (state_dict)
  memoire.npz   keys[M,NV] f32 · y_counts[M,NV] i32 · ok[M] f32 · n[M] i32 · last_seen[M] i64
  hist.npz      historique LOCAL (points APRES le parent uniquement)
  meta.json     identite : seeds + cycle + parent + parent_n + heldout_sha + code_sha + arch

Usage :
  python etat.py heldout <seed_eval> <n> <sortie.npz>
  python etat.py selftest
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import banc

FORMAT = "bebe-etat/1"           # v1 fige ; toute extension = bebe-etat/2


# ----------------------------------------------------------------- utilitaires
def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def arch_of(model) -> dict:
    enc = model.enc.layers
    return {"d": int(model.emb.embedding_dim), "nl": int(len(enc)),
            "heads": int(enc[0].self_attn.num_heads), "nv": int(model.head.out_features)}


def build_model(arch, seed_init=None, device="cpu"):
    if seed_init is not None:
        torch.manual_seed(int(seed_init))
    return banc.Bebe(d=int(arch["d"]), nl=int(arch["nl"]), heads=int(arch["heads"])).to(device)


# ----------------------------------------------------------------- memoire consolidee
class MemoireConsolidee:
    """Memoire kNN CONSOLIDEE PAR CLE — une cle = un contexte, pas une reponse.
    Interface identique a banc.MemoireKNN (add / consult) pour que banc.py puisse
    l'utiliser sans changer sa boucle. Format FIGE v1."""

    def __init__(self, k=5, cap=None):
        self.k = k
        self.cap = cap
        self.keys = np.zeros((0, banc.NV), np.float32)
        self.y_counts = np.zeros((0, banc.NV), np.int32)
        self.ok = np.zeros(0, np.float32)       # ok moyen par cle
        self.n = np.zeros(0, np.int32)          # nb d'ajouts sur la cle
        self.last_seen = np.zeros(0, np.int64)  # index d'interaction du dernier add
        self._ix = {}

    @staticmethod
    def vec(src):
        return banc.MemoireKNN.vec(src)

    def _rebuild_index(self):
        self._ix = {self.keys[r].tobytes(): r for r in range(len(self.keys))}

    def add(self, src, cible_idx, correct, step=0):
        v = self.vec(src)
        kb = v.tobytes()
        r = self._ix.get(kb)
        if r is None:
            self.keys = np.vstack([self.keys, v[None, :]])
            self.y_counts = np.vstack([self.y_counts, np.zeros((1, banc.NV), np.int32)])
            self.ok = np.append(self.ok, 0.0)
            self.n = np.append(self.n, 0)
            self.last_seen = np.append(self.last_seen, 0)
            r = len(self.keys) - 1
            self._ix[kb] = r
        self.y_counts[r, int(cible_idx)] += 1
        self.ok[r] = (self.ok[r] * self.n[r] + (1.0 if correct else 0.0)) / (self.n[r] + 1.0)
        self.n[r] += 1
        self.last_seen[r] = int(step)
        if self.cap and len(self.keys) > self.cap:
            self.prune()

    def consult(self, src):
        M = len(self.keys)
        if M < 3:
            return None, 0.0
        v = self.vec(src)
        d = np.linalg.norm(self.keys - v, axis=1)
        idx = np.argsort(d)[: min(self.k, M)]
        agg = self.y_counts[idx].sum(axis=0)
        tot = int(agg.sum())
        if tot == 0:
            return None, 0.0
        lab = int(agg.argmax())
        accord = agg[lab] / tot
        nn = int(self.n[idx].sum())
        fiab = float((self.ok[idx] * self.n[idx]).sum() / nn) if nn else 0.0
        return lab, float(fiab * accord)

    def prune(self):
        """Eviction LRU par last_seen quand cap depasse (POST-consolidation)."""
        if not self.cap or len(self.keys) <= self.cap:
            return
        keep = np.argsort(self.last_seen)[::-1][: self.cap]
        keep = np.sort(keep)
        self.keys = self.keys[keep]; self.y_counts = self.y_counts[keep]
        self.ok = self.ok[keep]; self.n = self.n[keep]; self.last_seen = self.last_seen[keep]
        self._rebuild_index()


# ----------------------------------------------------------------- held-out
def save_heldout(path, seed_eval, n):
    import random
    rng = random.Random(int(seed_eval))
    items = [banc.gen_item(rng) for _ in range(int(n))]
    np.savez(path, src=np.array([it[0] for it in items]),
             cible=np.array([it[1] for it in items]),
             seed_eval=np.array([int(seed_eval)]))
    return path


def load_heldout(path):
    z = np.load(path, allow_pickle=False)
    return list(zip([str(s) for s in z["src"]], [str(c) for c in z["cible"]]))


# ----------------------------------------------------------------- paquet
def save_bundle(path, model, mem, hist, meta):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "model.pt")
    np.savez(path / "memoire.npz",
             keys=mem.keys, y_counts=mem.y_counts,
             ok=mem.ok, n=mem.n, last_seen=mem.last_seen,
             k=np.array([int(mem.k)]))
    np.savez(path / "hist.npz", **{k: np.asarray(v, np.float32) for k, v in hist.items()})
    meta = dict(meta)
    meta["format"] = FORMAT
    meta.setdefault("created_utc", time.strftime("%Y-%m-%dT%H:%M:%S"))
    meta["arch"] = arch_of(model)
    (path / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_bundle(path, device="cpu"):
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    if meta.get("format") != FORMAT:
        raise ValueError(f"format inconnu : {meta.get('format')}")
    model = build_model(meta["arch"], device=device)
    model.load_state_dict(torch.load(path / "model.pt", map_location=device))
    z = np.load(path / "memoire.npz")
    mem = MemoireConsolidee(k=int(z["k"][0]))
    mem.keys, mem.y_counts = z["keys"], z["y_counts"]
    mem.ok, mem.n, mem.last_seen = z["ok"], z["n"], z["last_seen"]
    mem._rebuild_index()
    zz = np.load(path / "hist.npz")
    hist = {k: zz[k] for k in zz.files}
    return {"model": model, "mem": mem, "hist": hist, "meta": meta}


# ----------------------------------------------------------------- invariants durs
def check_invariants(bundles):
    metas = [b["meta"] for b in bundles]
    if len(metas) < 2:
        return True, ["(un seul fragment)"]
    raisons, fatal = [], False

    def dur(cond, msg):
        nonlocal fatal
        if cond:
            raisons.append(msg); fatal = True

    dur(len({m.get("heldout_sha256") for m in metas}) > 1,
        "heldout_sha256 divergent -> REFUSEE")
    dur(len({m.get("code_sha") for m in metas}) > 1,
        "code_sha divergent (le genome a bouge) -> REFUSEE")
    dur(len({json.dumps(m.get("arch"), sort_keys=True) for m in metas}) > 1,
        "arch differente -> REFUSEE")
    dur(len({m.get("parent") for m in metas}) > 1,
        "parent different (pas la meme famille) -> REFUSEE")
    dur(len({m.get("seeds", {}).get("init") for m in metas}) > 1,
        "seed_init divergent (pas de parent commun) -> REFUSEE")
    datas = [m.get("seeds", {}).get("data") for m in metas]
    dur(len(set(datas)) != len(datas),
        "seed_data dupliquee (fragments identiques) -> REFUSEE")
    if not raisons:
        raisons.append("OK : held-out, code_sha, arch, parent communs ; seeds data distinctes")
    return (not fatal), raisons


# ----------------------------------------------------------------- selftest
def _selftest():
    torch.manual_seed(1234)
    model = banc.Bebe()
    mem = MemoireConsolidee(k=5)
    for i, (s, c) in enumerate([("abc", "a"), ("abc", "a"), ("hijk", "h")]):
        mem.add(s, banc.VOCAB.get(c, 0), True, step=i)
    hist = {"conf": [0.5, 0.6], "ok": [0.0, 1.0]}
    meta = {"fragment": "selftest", "parent": "root", "cycle": 0, "parent_n": 0,
            "seeds": {"init": 1234, "data": 5678, "eval": 999},
            "heldout_sha256": "deadbeef", "code_sha": "none"}
    d = Path(__file__).resolve().parent / "sorties_etat_test"
    save_bundle(d, model, mem, hist, meta)
    b = load_bundle(d)
    assert arch_of(b["model"]) == arch_of(model)
    assert len(b["mem"].keys) == 2, f"consolidation ratee : {len(b['mem'].keys)} cles (2 attendues)"
    assert int(b["mem"].n[0]) == 2, "compteur de la cle 'abc' devrait valoir 2"
    print("selftest OK ->", d, "| cles:", len(b["mem"].keys), "| n[0]:", int(b["mem"].n[0]))


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "heldout":
        save_heldout(sys.argv[4], int(sys.argv[2]), int(sys.argv[3]))
        print("held-out ecrit :", sys.argv[4], "| sha", sha256_file(sys.argv[4])[:12])
    elif len(sys.argv) >= 2 and sys.argv[1] == "selftest":
        _selftest()
    else:
        print(__doc__)
