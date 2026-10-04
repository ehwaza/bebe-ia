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


def charger_donnees(source, seed_data, cache=None, reserve_frac=0.0):
    """Generateur INFINI d'interactions (src, cible). Reproductible.

    reserve_frac > 0 : plan B (zero fuite par construction) -- la queue du
    corpus filtre est reservee au heldout ; le train ne voit que la tete et
    la marche modulo ne peut JAMAIS l'atteindre. Defaut 0.0 = comportement
    identique aux runs existants (zéro regression). Doit etre genere par le
    generateur wiki_heldout avec le MEME cache (sinon cut different).
    """
    if source == "synthetique":
        rng = random.Random(seed_data)
        pool = [banc.gen_item(rng) for _ in range(1200)]  # le monde revient
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


# ------------------------------------------------------------------ boucle
def boucle_entrainement(n, out_dir, seed, seed_data, source="synthetique",
                        cache=None, heldout_path=None, budget_s=None,
                        eval_every=1000, ckpt_every=25000, device=None,
                        amp=False, lr=3e-4, parent_dir=None, verbose=True,
                        reserve_frac=0.0):
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
    gen = charger_donnees(source, seed_data, cache, reserve_frac)

    t0, n_fait, ms_lst = time.time(), 0, []
    for i in range(int(n)):
        src, cible = next(gen)
        y = banc.VOCAB.get(cible, -1)
        if y < 0:
            continue
        ids = banc.to_ids(src).unsqueeze(0).to(device)

        model.train()
        t_iter = time.time()
        with torch.amp.autocast("cuda", enabled=amp and device.startswith("cuda")):
            logit = model(ids)
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
                    ms_lst, source)

        if budget_s and (time.time() - t0) > float(budget_s):
            if verbose:
                print(f"budget temps atteint ({budget_s}s) a {n_fait} interactions")
            break

    res = _sauver(out_dir, model, mem, hist, parent_meta, cycle,
                  parent_n, n_fait, heldout_path, seed, seed_data,
                  ms_lst, source)
    return res


def evaluer_rapide(model, mem, items, device="cpu"):
    from bebe.fusion import evaluer
    return evaluer(model, mem, items, device)["ece"]


def _sauver(out_dir, model, mem, hist, parent_meta, cycle, parent_n,
            n_fait, heldout_path, seed, seed_data, ms_lst, source):
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
        "metrics": {
            "ece": float(hist["ece"][-1]) if hist["ece"] else float("nan"),
            "ms_per_iter": ms_moy,
            "items_per_s": 1000.0 / ms_moy if ms_moy == ms_moy else float("nan"),
        },
    }
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
    ap.add_argument("--budget-seconds", type=float, default=None)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--ckpt-every", type=int, default=25000)
    ap.add_argument("--reprendre", default=None, help="dossier etat/ a continuer")
    ap.add_argument("--amp", action="store_true", help="mixed precision GPU")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    seed_data = a.seed_data if a.seed_data is not None \
        else int(time.time()) ^ (os.getpid() << 8)
    r = boucle_entrainement(
        n=a.n, out_dir=a.out, seed=a.seed, seed_data=seed_data,
        source=a.source, cache=a.cache, heldout_path=a.heldout,
        budget_s=a.budget_seconds, eval_every=a.eval_every,
        ckpt_every=a.ckpt_every, device=a.device, amp=a.amp,
        reserve_frac=a.reserve_frac,
        parent_dir=(Path(a.reprendre) if a.reprendre else None))
    print(json.dumps(r, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
