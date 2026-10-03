# -*- coding: utf-8 -*-
"""bebe/fusion.py — FUSION PAR CONFIANCE de deux etats du bebe. (nouveau)

Principe (Mathieu, 04/10) :
  1. On evalue CHAQUE etat sur UN MEME echantillon commun (le held-out fige).
  2. On pondere les poids par la confiance mesuree : meilleur ECE = plus de poids.
  3. On fusionne poids (moyenne ponderee), memoire kNN (consolidation par cle),
     historique (concatenation des locaux).
  4. On sauvegarde l'etat fusionne (format bebe-etat/1, C1-C5).

INVARIANTS PROPRE A CE MODULE (fratrie, meme held-out) :
  - heldout_sha256 IDENTIQUE  -> sinon REFUS (on ne mesure pas la meme chose)
  - code_sha IDENTIQUE        -> sinon REFUS (le genome a bouge)
  - arch IDENTIQUE            -> sinon REFUS
  - seed_data DIFFERENTE      -> sinon REFUS (fragments clones)
  Parent/seed_init peuvent differer (fratrie issus de graines distinctes) :
  c'est exactement le cas local vs GitHub, la fusion est faite pour ca.

Usage : python bebe/fusion.py <etat1> <etat2> --heldout heldout.npz --out dossier
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import banc  # noqa: E402
import etat  # noqa: E402

EPS = 1e-4


# ------------------------------------------------------------------ evaluation
def evaluer(model, mem, items, device="cpu"):
    """Fidele au bras A du banc : conf = Pmax du softmax brut, boost kNN > 0.5."""
    model.eval()
    confs, oks = [], []
    for src, cible in items:
        ids = banc.to_ids(src).unsqueeze(0).to(device)
        with torch.no_grad():
            p = torch.softmax(model(ids), 1)[0]
        conf, pred = float(p.max()), int(p.argmax())
        if mem is not None:
            maj, sk = mem.consult(src)
            if maj is not None and sk > 0.5:
                conf = min(0.99, max(conf, sk))
        confs.append(conf)
        oks.append(1.0 if pred == banc.VOCAB.get(cible, -1) else 0.0)
    confs, oks = np.asarray(confs), np.asarray(oks)
    return {"ece": banc.ece_score(confs, oks), "acc": float(oks.mean()),
            "conf_moy": float(confs.mean()), "n": len(oks)}


# ------------------------------------------------------------------ invariants fratrie
def check_fraternel(metas):
    raisons, fatal = [], False

    def dur(cond, msg):
        nonlocal fatal
        if cond:
            raisons.append(msg)
            fatal = True

    dur(len({m.get("heldout_sha256") for m in metas}) > 1,
        "heldout_sha256 divergent -> REFUSEE (pas la meme regle du jeu)")
    dur(len({m.get("code_sha") for m in metas}) > 1,
        "code_sha divergent -> REFUSEE (le genome a bouge)")
    dur(len({json.dumps(m.get("arch"), sort_keys=True) for m in metas}) > 1,
        "arch differente -> REFUSEE")
    datas = [m.get("seeds", {}).get("data") for m in metas]
    dur(len(set(datas)) != len(datas),
        "seed_data dupliquee -> REFUSEE (fragments clones)")
    if not fatal:
        raisons.append("OK : held-out, code_sha, arch communs ; seed_data distinctes")
    return (not fatal), raisons


# ------------------------------------------------------------------ memoire : consolidation
def merge_memoires(mems, cap=banc.CAP_MEMOIRE):
    """Union par cle + SOMME des compteurs — meme voie que add()."""
    keys, yc, ok_num, n, ls, ix = [], [], [], [], [], {}
    for mem in mems:
        for r in range(len(mem.keys)):
            kb = mem.keys[r].tobytes()
            j = ix.get(kb)
            if j is None:
                ix[kb] = len(keys)
                keys.append(mem.keys[r])
                yc.append(mem.y_counts[r].astype(np.int32).copy())
                ok_num.append(float(mem.ok[r]) * int(mem.n[r]))
                n.append(int(mem.n[r]))
                ls.append(int(mem.last_seen[r]))
            else:
                yc[j] = yc[j] + mem.y_counts[r]
                ok_num[j] += float(mem.ok[r]) * int(mem.n[r])
                n[j] += int(mem.n[r])
                ls[j] = max(ls[j], int(mem.last_seen[r]))
    out = etat.MemoireConsolidee(k=mems[0].k, cap=cap)
    out.keys = np.array(keys, np.float32)
    out.y_counts = np.array(yc, np.int32)
    out.n = np.array(n, np.int32)
    out.ok = np.array([ok_num[i] / n[i] if n[i] else 0.0
                       for i in range(len(n))], np.float32)
    out.last_seen = np.array(ls, np.int64)
    out._rebuild_index()
    return out


def concat_hist(hists):
    out = {}
    for k in set().union(*[h.keys() for h in hists]):
        parts = [np.asarray(h[k], np.float32) for h in hists if k in h]
        out[k] = np.concatenate(parts) if parts else np.zeros(0, np.float32)
    return out


# ------------------------------------------------------------------ fusion par confiance
def fusionner_confiance(chemins, heldout_path, out_dir=None, device="cpu"):
    """Fusion ponderee par la confiance (meilleur ECE = plus de poids)."""
    items = etat.load_heldout(heldout_path)
    bundles = [etat.load_bundle(p, device) for p in chemins]
    metas = [b["meta"] for b in bundles]
    ok_inv, raisons = check_fraternel(metas)
    if not ok_inv:
        return {"ok": False, "raisons": raisons}

    # 1. evaluation commune -> ECE par fragment
    evals = [evaluer(b["model"], b["mem"], items, device) for b in bundles]
    eces = np.array([e["ece"] for e in evals], float)

    # 2. poids par confiance : meilleur ECE = poids plus grand
    w = (1.0 / (eces + EPS))
    w = w / w.sum()

    # 3. moyenne ponderee des poids
    sds = [b["model"].state_dict() for b in bundles]
    fused_sd = {k: sum(float(w[i]) * sds[i][k].float()
                       for i in range(len(sds))) for k in sds[0]}
    model = etat.build_model(etat.arch_of(bundles[0]["model"]), device=device)
    model.load_state_dict(fused_sd)

    mem = merge_memoires([b["mem"] for b in bundles])
    hist = concat_hist([b["hist"] for b in bundles])

    # 4. meta C1-C5 : parent C4, comptes sans double compte
    #    regle : n_interactions = parent_n + SOMME des n_new (le parent
    #    partage compte une seule fois).
    parent_n = max(int(m.get("parent_n", 0)) for m in metas)
    n_new = sum(int(m.get("n_new", 0)) for m in metas)
    p0 = metas[0]
    meta = {
        "fragment": "fusion-confiance",
        "parent": f'{p0.get("fragment")}@{p0.get("code_sha")}#{p0.get("cycle")}',
        "cycle": max(int(m.get("cycle", 0)) for m in metas) + 1,
        "parent_n": parent_n,
        "n_new": n_new,
        "n_interactions": parent_n + n_new,
        "seeds": {"init": None, "data": -1, "eval": p0.get("seeds", {}).get("eval")},
        "heldout_sha256": p0.get("heldout_sha256"),
        "code_sha": p0.get("code_sha"),
        "metrics": {"ece": float(eces.min())},
        "fusion": {"methode": "confiance-ece", "poids": [float(x) for x in w],
                   "eces": [float(x) for x in eces]},
    }

    res = {"ok": True, "raisons": raisons,
           "eces_fragments": [float(x) for x in eces],
           "poids": [float(x) for x in w],
           "evals": evals}

    if out_dir:
        etat.save_bundle(out_dir, model, mem, hist, meta)
        res["out"] = str(out_dir)
        (Path(out_dir) / "fusion.json").write_text(
            json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    res["model"], res["mem"], res["hist"] = model, mem, hist
    return res


def main():
    ap = argparse.ArgumentParser(description="Fusion par confiance de deux etats")
    ap.add_argument("etats", nargs="+", help="dossiers de paquets (2+)")
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    r = fusionner_confiance(a.etats, a.heldout, a.out, a.device)
    print(json.dumps({k: v for k, v in r.items()
                      if k not in ("model", "mem", "hist")},
                     indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
