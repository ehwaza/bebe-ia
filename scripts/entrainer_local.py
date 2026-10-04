# -*- coding: utf-8 -*-
"""scripts/entrainer_local.py — cycle d'entrainement LOCAL du bebe.

Le BEBE apprend EN LIGNE, SANS SUPERVISION DE CONFIANCE :
  1 interaction = 1 contexte -> 1 prediction -> 1 feedback 0/1 -> 1 step SGD.
Sa confiance est le Pmax du softmax brut (+ boost memoire kNN). Personne
ne lui dit s'il DOIT etre confiant : il ne voit que juste/faux, et doit
apprendre seul a savoir quand il ne sait pas.

Sources de donnees :
  synthetique : pool fini d'items (le monde revient)          [defaut local]
  texte       : corpus .txt local/Drive -> fenetres glissantes
  wikipedia   : corpus Wikipedia via `datasets` (cache local/Drive)

Sorties dans --out :
  etat/          paquet C1-C5 (model.pt, memoire.npz, hist.npz, meta.json)
  hist.npz       trace complete (conf, ok, ece)
  resultats.json metriques finales + ms/iter mesure (pour caler les jobs)

Usage :
  python scripts/entrainer_local.py --n 100000 --out resultats/local --seed 1234
  python scripts/entrainer_local.py --reprendre resultats/local --n 50000
  python scripts/entrainer_local.py --miel fold.npz --regles rep ...   [Phase 2]

PHASE 2 (PROTOCOLE_PHASE2.md, pre-enregistre 04/10) :
  --miel <npz> charge le miel de ruche FIXE (t=0, jamais mis a jour = anti-
  auto-leak P2) et module la cible d'entrainement (protocole §3, knob-free) :
      si sk > 0.5 : target = sk*onehot(maj) + (1-sk)*onehot(y)
      sinon       : target = onehot(y)  [chemin STRICT identique a Phase 1]
  Sans --miel, le chemin de calcul est byte-identique aux runs existants.
  Telemetrie P6 : meta.miel.taux_activation (sinon un nul est mecanique).
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import banc  # noqa: E402
import etat  # noqa: E402
from bebe.metriques import resume_metriques  # noqa: E402

L_FEN = 24  # longueur de fenetre pour les sources texte


# ------------------------------------------------------------------ donnees
def _charger_wikipedia(cache, n_articles=2000):
    """Corpus Wikipedia -> texte brut, cache dans `cache` (Drive possible)."""
    cache = Path(cache)
    brut = cache / "wiki_cache.txt"
    if brut.exists():
        return brut.read_text(encoding="utf-8", errors="ignore")
    from datasets import load_dataset  # dependance optionnelle
    ds = load_dataset("wikimedia/wikipedia", "20231101.fr",
                      split="train", streaming=True)
    morceaux = []
    for i, art in enumerate(ds):
        if i >= n_articles:
            break
        morceaux.append(art["text"])
    cache.mkdir(parents=True, exist_ok=True)
    brut.write_text("\n".join(morceaux), encoding="utf-8")
    return "\n".join(morceaux)


def _gen_filtre(rng, regles):
    """Item de banc.gen_item RESTREINT a un sous-ensemble de regles.

    Même algorithme que fragment.py (rejet) : meme rng -> meme pool,
    local/GitHub/Colab partagent la meme distribution. Defaut None = intact.
    """
    if not regles:
        return banc.gen_item(rng)
    while True:
        it = banc.gen_item(rng)
        if it[2] in regles:
            return it


def charger_donnees(source, seed_data, cache=None, reserve_frac=0.0,
                    regles=None, central_spec=None):
    """Generateur INFINI d'interactions (src, cible). Reproductible.

    reserve_frac > 0 : plan B (zero fuite par construction) -- la queue du
    corpus filtre est reservee au heldout ; le train ne voit que la tete et
    la marche modulo ne peut JAMAIS l'atteindre. Defaut 0.0 = comportement
    identique aux runs existants (zero regression). Doit etre genere par le
    generateur wiki_heldout avec le MEME cache (sinon cut different).
    regles : sous-ensemble de banc.gen_item restreint (distribution par shard).
    central_spec : JSON [{data_seed,regles,pool},...] -> pool = concat des
    shards (LE CONTROLE central, meme multiset que l'union des fragments).
    """
    if source == "synthetique":
        if isinstance(regles, str):
            regles = set(r.strip() for r in regles.split(",") if r.strip()) or None
        if central_spec:
            spec = json.loads(central_spec)
            pool = []
            for s in spec:
                rr = random.Random(int(s["data_seed"]))
                rg = set(x.strip() for x in s.get("regles", "").split(",")
                         if x.strip()) or None
                pool += [_gen_filtre(rr, rg) for _ in range(int(s.get("pool", 300)))]
        else:
            rng = random.Random(seed_data)
            pool = [_gen_filtre(rng, regles) for _ in range(1200)]
        i = 0
        while True:
            src, cible, _ = pool[i % len(pool)]
            i += 1
            yield src, cible
    if source == "wikipedia":
        texte = _charger_wikipedia(cache or (ROOT / "cache"))
    else:  # texte
        texte = Path(cache).read_text(encoding="utf-8", errors="ignore")
    texte = "".join(c for c in texte.lower() if c in banc.VOCAB)
    if reserve_frac > 0:  # coupe APRES filtre VOCAB (meme ordre que le generateur)
        texte = texte[:int(len(texte) * (1 - reserve_frac))]
    if len(texte) < L_FEN + 2:
        raise ValueError("corpus trop petit pour le source " + source)
    rng = random.Random(seed_data)
    i = rng.randrange(len(texte) - L_FEN - 1)
    while True:
        fen = texte[i:i + L_FEN + 1]
        yield fen[:-1], fen[-1]
        i = (i + 1 + (seed_data % 7)) % (len(texte) - L_FEN - 1)


# ------------------------------------------------------------------ PHASE 2
def cible_miel(nv, y, maj, sk):
    """Cible miel — PROTOCOLE_PHASE2.md §3 (pre-enregistre, knob-free).

    sk > 0.5 : target = sk*onehot(maj) + (1-sk)*onehot(y)
    sinon    : None (l'appelant garde la cible standard = onehot(y), chemin
    strictement identique a Phase 1).
    Le seul poids est sk lui-meme ; le seuil 0.5 est deja celui du mode DUR
    de la spec nectar v1. si maj == y, la masse se recombine a 1.0 sur y.
    Retourne un vecteur np.float32 de longueur nv.
    """
    if maj is None or not (sk > 0.5):
        return None
    t = np.zeros(int(nv), np.float32)
    t[int(maj)] += np.float32(sk)
    t[int(y)] += np.float32(1.0) - np.float32(sk)
    return t


def _charger_miel(path):
    """Miel de ruche FIXE (Phase 2) : npz {keys, y_counts, n, ok}.

    Contrat P2 : charge t=0, JAMAIS mis a jour pendant le run (jamais .add).
    Consult = banc.MemoireConsolidee, keys cast f32 — LE MEME cast que le
    harnais d'eval des deux cotes -> sk identiques train/eval.
    """
    z = np.load(path)
    m = banc.MemoireConsolidee(k=5, cap=None)
    m.keys = np.asarray(z["keys"], np.float32)
    m.y_counts = np.asarray(z["y_counts"], np.int32)
    m.n = np.asarray(z["n"], np.int32)
    m.ok = np.asarray(z["ok"], np.float32)
    return m, int(m.keys.shape[0])


def _resume_miel(path, sha, m, n_miel, n_act, sk_vals, code_entre):
    """Telemetrie P6 : taux d'activation de la cible-miel (obligatoire —
    sans elle, un nul en (A) est mecanique et non interpretable)."""
    a = np.asarray(sk_vals, np.float64)
    d = {"fichier": str(path), "sha256": sha, "m": int(m),
         "consultes": int(n_miel), "actifs": int(n_act),
         "taux_activation": (float(n_act) / n_miel) if n_miel else float("nan"),
         "code_sha_entre": code_entre}
    if a.size:
        d.update({"sk_moy": float(a.mean()),
                  "sk_p50": float(np.percentile(a, 50)),
                  "sk_p90": float(np.percentile(a, 90)),
                  "sk_p99": float(np.percentile(a, 99)),
                  "sk_max": float(a.max())})
    else:
        d.update({k: float("nan") for k in
                  ("sk_moy", "sk_p50", "sk_p90", "sk_p99", "sk_max")})
    return d


# ------------------------------------------------------------------ boucle
def boucle_entrainement(n, out_dir, seed, seed_data, source="synthetique",
                        cache=None, heldout_path=None, budget_s=None,
                        eval_every=1000, ckpt_every=25000, device=None,
                        amp=False, lr=3e-4, parent_dir=None, verbose=True,
                        reserve_frac=0.0, regles=None, central_spec=None,
                        miel_path=None):
    """Coeur d'entrainement partage (local, GitHub, Colab, multi-seeds)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    heldout_path = heldout_path or (ROOT / "heldout.npz")
    heldout = etat.load_heldout(heldout_path)[:400]  # echantillon de mesure

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # --- reprise ou naissance
    parent_meta = None
    if parent_dir and (Path(parent_dir) / "meta.json").exists():
        b = etat.load_bundle(parent_dir, device)
        model, mem = b["model"], b["mem"]
        hist = {k: list(v) for k, v in b["hist"].items()}
        parent_meta = b["meta"]
        cycle = int(parent_meta.get("cycle", 0)) + 1
        parent_n = int(parent_meta.get("parent_n", 0)) + int(parent_meta.get("n_new", 0))
    else:
        model = etat.build_model({"d": 128, "nl": 2, "heads": 4},
                                 seed_init=seed, device=device)
        mem = etat.MemoireConsolidee(k=5, cap=banc.CAP_MEMOIRE)
        hist = {"conf": [], "ok": [], "ece": []}
        cycle, parent_n = 0, 0

    opt = torch.optim.SGD(model.parameters(), lr=lr)
    lossf = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.startswith("cuda"))
    if isinstance(regles, str):  # normalise pour le pool ET la trace meta
        regles = set(r.strip() for r in regles.split(",") if r.strip()) or None
    gen = charger_donnees(source, seed_data, cache, reserve_frac,
                          regles, central_spec)

    # --- miel de ruche FIXE (Phase 2) : charge t=0, jamais mis a jour (P2).
    #     Sans --miel : miel=None et le chemin est identique a Phase 1.
    miel, miel_m = None, 0
    miel_sha = None
    if miel_path:
        miel, miel_m = _charger_miel(miel_path)
        miel_sha = etat.sha256_file(Path(miel_path))
    code_entre = etat.sha256_file(Path(__file__))[:12]
    n_miel, n_act, sk_vals = 0, 0, []

    t0, n_fait, ms_lst = time.time(), 0, []
    for i in range(int(n)):
        src, cible = next(gen)
        y = banc.VOCAB.get(cible, -1)
        if y < 0:
            continue
        ids = banc.to_ids(src).unsqueeze(0).to(device)

        # Phase 2 (protocole §3) : consult du miel fixe AVANT (le miel n'est
        # jamais modifie ; la regle consult-avant-add est donc tenue).
        tgt_m = None
        if miel is not None:
            maj_m, sk_m = miel.consult(src)
            n_miel += 1
            sk_vals.append(float(sk_m))
            if maj_m is not None and sk_m > 0.5:
                n_act += 1
                tgt_m = cible_miel(banc.NV, y, maj_m, sk_m)

        model.train()
        t_iter = time.time()
        with torch.amp.autocast("cuda", enabled=amp and device.startswith("cuda")):
            logit = model(ids)
            if tgt_m is not None:
                loss = lossf(logit, torch.from_numpy(tgt_m).unsqueeze(0).to(device))
            else:
                loss = lossf(logit, torch.tensor([y], device=device))
        p = torch.softmax(logit.detach().float(), 1)[0]
        conf, pred = float(p.max()), int(p.argmax())
        maj, sk = mem.consult(src)
        if maj is not None and sk > 0.5:
            conf = min(0.99, max(conf, sk))       # la memoire conforte
        mem.add(src, pred, pred == y, step=parent_n + i)

        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        ms_lst.append((time.time() - t_iter) * 1000.0)
        n_fait += 1

        hist["conf"].append(conf)
        hist["ok"].append(1.0 if pred == y else 0.0)

        if (i + 1) % eval_every == 0 or i == int(n) - 1:
            e = evaluer_rapide(model, mem, heldout, device)
            hist["ece"].append(e)
            if verbose:
                ms = float(np.mean(ms_lst[-eval_every:]))
                print(f"[{i+1:8d}/{n}] ece_heldout={e:.3f} "
                      f"acc={np.mean(hist['ok'][-eval_every:]):.2f} "
                      f"({ms:.1f} ms/it)", flush=True)

        if ckpt_every and (i + 1) % ckpt_every == 0:
            _sauver(out_dir, model, mem, hist, parent_meta, cycle,
                    parent_n, n_fait, heldout_path, seed, seed_data,
                    ms_lst, source, regles,
                    miel_meta=(_resume_miel(miel_path, miel_sha, miel_m,
                                            n_miel, n_act, sk_vals, code_entre)
                               if miel_path else None))

        if budget_s and (time.time() - t0) > float(budget_s):
            if verbose:
                print(f"budget temps atteint ({budget_s}s) a {n_fait} interactions")
            break

    if verbose and miel_path and n_miel:
        print(f"miel phase2 : activation cible {n_act}/{n_miel} "
              f"({100.0 * n_act / n_miel:.1f}%) sk>0.5", flush=True)

    res = _sauver(out_dir, model, mem, hist, parent_meta, cycle,
                  parent_n, n_fait, heldout_path, seed, seed_data,
                  ms_lst, source, regles,
                  miel_meta=(_resume_miel(miel_path, miel_sha, miel_m,
                                          n_miel, n_act, sk_vals, code_entre)
                             if miel_path else None))
    return res


def evaluer_rapide(model, mem, items, device="cpu"):
    from bebe.fusion import evaluer
    return evaluer(model, mem, items, device)["ece"]


def _sauver(out_dir, model, mem, hist, parent_meta, cycle, parent_n,
            n_fait, heldout_path, seed, seed_data, ms_lst, source, regles,
            miel_meta=None):
    ms_moy = float(np.mean(ms_lst)) if ms_lst else float("nan")
    code_sha = etat.sha256_file(ROOT / "banc.py")[:12]
    parent = ("root" if parent_meta is None else
              f'{parent_meta.get("fragment")}@{parent_meta.get("code_sha")}'
              f'#{parent_meta.get("cycle")}')
    meta = {
        "fragment": f"local-{seed_data % 100000:05d}",
        "parent": parent,
        "cycle": int(cycle),
        "parent_n": int(parent_n),
        "n_new": int(n_fait),
        "n_interactions": int(parent_n + n_fait),
        "seeds": {"init": int(seed), "data": int(seed_data), "eval": 999},
        "heldout_sha256": etat.sha256_file(heldout_path),
        "code_sha": code_sha,
        "source": source,
        "regles": (sorted(regles) if regles else None),
        "metrics": {
            "ece": float(hist["ece"][-1]) if hist["ece"] else float("nan"),
            "ms_per_iter": ms_moy,
            "items_per_s": 1000.0 / ms_moy if ms_moy == ms_moy else float("nan"),
        },
    }
    if miel_meta:  # Phase 2 seulement : absent des runs sans --miel
        meta["miel"] = miel_meta
    etat.save_bundle(out_dir / "etat", model, mem, hist, meta)
    np.savez(out_dir / "hist.npz",
             **{k: np.asarray(v, np.float32) for k, v in hist.items()})
    resume = resume_metriques(hist["conf"], hist["ok"])
    resume.update({"ms_per_iter": ms_moy, "n_new": int(n_fait),
                   "cycle": int(cycle), "source": source})
    (out_dir / "resultats.json").write_text(
        json.dumps({"resume": resume, "meta": meta, "trace_ece": hist["ece"]},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    return resume


# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--out", default=str(ROOT / "resultats" / "local"))
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--seed-data", type=int, default=None,
                    help="graine des donnees (DOIT etre unique par fragment)")
    ap.add_argument("--source", default="synthetique",
                    choices=["synthetique", "texte", "wikipedia"])
    ap.add_argument("--cache", default=None, help="cache corpus (Drive possible)")
    ap.add_argument("--heldout", default=str(ROOT / "heldout.npz"))
    ap.add_argument("--reserve-frac", type=float, default=0.0,
                    help="queue du corpus reservee au heldout (0 = off)")
    ap.add_argument("--regles", default=None,
                    help="regles de gen_item autorisees (ex: arith,rep). Vide = toutes")
    ap.add_argument("--central-spec", default=None,
                    help="JSON [{data_seed,regles,pool},...] -> concat des shards (LE CONTROLE)")
    ap.add_argument("--budget-seconds", type=float, default=None)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--ckpt-every", type=int, default=25000)
    ap.add_argument("--reprendre", default=None, help="dossier etat/ a continuer")
    ap.add_argument("--amp", action="store_true", help="mixed precision GPU")
    ap.add_argument("--device", default=None)
    ap.add_argument("--miel", default=None,
                    help="npz du miel de ruche FIXE {keys,y_counts,n,ok} : "
                         "active la cible miel Phase 2 (PROTOCOLE §3)")
    a = ap.parse_args()
    seed_data = a.seed_data if a.seed_data is not None \
        else int(time.time()) ^ (os.getpid() << 8)
    r = boucle_entrainement(
        n=a.n, out_dir=a.out, seed=a.seed, seed_data=seed_data,
        source=a.source, cache=a.cache, heldout_path=a.heldout,
        budget_s=a.budget_seconds, eval_every=a.eval_every,
        ckpt_every=a.ckpt_every, device=a.device, amp=a.amp,
        reserve_frac=a.reserve_frac, regles=a.regles,
        central_spec=a.central_spec, miel_path=a.miel,
        parent_dir=(Path(a.reprendre) if a.reprendre else None))
    print(json.dumps(r, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
