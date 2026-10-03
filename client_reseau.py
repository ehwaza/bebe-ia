# -*- coding: utf-8 -*-
"""CLIENT RESEAU — F:\becvide\client_reseau.py

NOEUD VOLONTAIRE (pas un botnet) : telecharge des pages du web de
SURFACE dans un ocean CONTROLE (API REST officielle Wikipedia uniquement),
les convertit en items mesurables, et mesure la calibration du bebe
exactement comme le banc local.

La verite ne vient plus de gen_item synthetique : elle vient du monde.
Item = (contexte de 24 c. reels -> caractere suivant reel). La page
repond elle-meme. ZLLM, zéro annotation humaine.

Reutilise l'instrument : Bebe, MemoireKNN, ece_score, risk_coverage
sont importes de banc.py — un seul instrument, deux mondes.

Politesse (contrat technique, regles du 1% ) :
  - un seul hote : en.wikipedia.org (API concue pour les clients)
  - 1 requete / --delay s (defaut 1.5), timeout 10 s, UA identitaire
  - chaque page est mise en cache : un telechargement = UNE fois
  - --local : aucun appel reseau, cache seulement
  - echec reseau -> on travaille sur le cache, pas d'insistance

Usage :
  python client_reseau.py --pages 30 --items 200
  python client_reseau.py --local            # sans reseau
  python client_reseau.py --pages 1 --items 80   # fumee rapide
"""
import argparse
import json
import random
import sys
import time
import unicodedata
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from banc import (ALPHABET, VOCAB, NV, DEVICE, Bebe, to_ids,
                  ece_score, risk_coverage, MemoireConsolidee, CAP_MEMOIRE)

SEED = 1234
HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
OUT = HERE / "sorties_reseau"
OUT.mkdir(exist_ok=True)

API = "https://en.wikipedia.org/api/rest_v1/page/random/summary"
UA = "becvide-node/0.1 (banc de calibration volontaire, 1 req/1.5s)"


# ------------------------------------------------------------------ reseau
def fetch_page(timeout=10):
    """Une page au hasard via l'API REST officielle. (titre, texte)."""
    req = urllib.request.Request(API, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    return data.get("title", "?"), data.get("extract", "")


def ensure_cache(n_pages, delay, local):
    """Retourne <= n_pages fichiers .txt. Cree le cache si besoin."""
    CACHE.mkdir(exist_ok=True)

    def existants():
        return sorted(CACHE.glob("*.txt"))

    if local:
        fichiers = existants()[:n_pages]
        if not fichiers:
            sys.exit("--local : cache vide, rien a mesurer.")
        print(f"mode local : {len(fichiers)} page(s) depuis le cache")
        return fichiers

    cible = n_pages
    essais = 0
    while len(existants()) < cible and essais < cible + 3:
        essais += 1
        try:
            titre, texte = fetch_page()
        except Exception as e:                       # reseau KO -> cache
            print(f"fetch KO ({e}) — on reste sur le cache", flush=True)
            time.sleep(delay * 2)
            continue
        if len(texte) < 200:
            continue
        slug = "".join(c for c in titre.lower() if c.isalnum())[:40] or "page"
        idx = len(existants())
        (CACHE / f"{idx:04d}_{slug}.txt").write_text(
            titre + "\n\n" + texte, encoding="utf-8")
        print(f"fetch OK : {titre} ({len(texte)} c.)", flush=True)
        time.sleep(delay)

    fichiers = existants()[:cible]
    if not fichiers:
        sys.exit("aucune page disponible (reseau KO, cache vide).")
    if len(fichiers) < cible:
        print(f"attention : {len(fichiers)}/{cible} pages (reseau partiel)")
    return fichiers


# ------------------------------------------------------------------ items
def norm(texte):
    """Texte reel -> alphabet du banc (accents ropes, minuscules)."""
    t = unicodedata.normalize("NFKD", texte).encode("ascii", "ignore").decode()
    return "".join(c for c in t.lower() if c in ALPHABET)


def gen_rnd(rng):
    """Item sans verite (comme banc) : confiance honnete basse."""
    src = "".join(rng.choice(ALPHABET[:26]) for _ in range(rng.randint(4, 9)))
    return src, rng.choice(ALPHABET[:26]), "rnd"


def items_depuis_pages(fichiers, limite, frac_rnd, rng):
    """Pages -> items (contexte 24 c., cible = caractere suivant reel)."""
    items = []
    titres = []
    for f in fichiers:
        brut = f.read_text(encoding="utf-8")
        titres.append(brut.splitlines()[0][:80])
        s = norm(brut)
        fen = [s[i:i + 25] for i in range(len(s) - 24)]
        fen = [w for w in fen if w[24] in VOCAB]
        if len(fen) > limite:                 # sous-echantillonnage egal
            pas = max(1, len(fen) // limite)
            fen = fen[::pas][:limite]
        items += [(w[:24], w[24], "web") for w in fen]

    if frac_rnd > 0 and items:
        k = max(2, int(1 / frac_rnd))
        mixe = []
        for i, it in enumerate(items):
            mixe.append(it)
            if (i + 1) % k == 0:
                mixe.append(gen_rnd(rng))
        items = mixe
    return items, titres


# ------------------------------------------------------------------ bras B
def entrainer_B(items, epochs=6):
    """Reference supervisee : batch sur les 3/4 premiers + temperature."""
    torch.manual_seed(SEED)
    modelB = Bebe().to(DEVICE)
    optB = torch.optim.Adam(modelB.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    split = int(len(items) * 0.75)
    modelB.train()
    for _ in range(epochs):
        for src, cible, _ in items[:split]:
            ids = to_ids(src).unsqueeze(0).to(DEVICE)
            logit = modelB(ids)
            optB.zero_grad()
            lossf(logit, torch.tensor([VOCAB[cible]], device=DEVICE)).backward()
            optB.step()

    temp = nn.Parameter(torch.ones(1, device=DEVICE))
    optT = torch.optim.LBFGS([temp], lr=0.1, max_iter=30)
    modelB.eval()

    def calib():
        tot = 0.0
        for src, cible, _ in items[split:split + 400]:
            ids = to_ids(src).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logit = modelB(ids) / temp.clamp(0.05, 10)
            tot += lossf(logit, torch.tensor([VOCAB[cible]], device=DEVICE))
        return tot

    try:
        optT.step(calib)
    except Exception:
        pass
    T = float(temp.detach().clamp(0.05, 10).item())
    return modelB, T


# ------------------------------------------------------------------ boucle
def mesurer(items, modelB, T, log_every=100):
    """Boucle identique a banc.run_bras : A en-ligne, B fige, C 0.5."""
    rng = random.Random(SEED)
    torch.manual_seed(SEED + 1)
    modelA = Bebe().to(DEVICE)
    optA = torch.optim.SGD(modelA.parameters(), lr=3e-4)
    mem = MemoireConsolidee(k=5, cap=CAP_MEMOIRE)
    lossf = nn.CrossEntropyLoss()

    hist = {"A": {"conf": [], "ok": [], "ece": []},
            "B": {"conf": [], "ok": [], "ece": []},
            "C": {"conf": [], "ok": [], "ece": []}}
    n = len(items)
    t0 = time.time()
    for i in range(n):
        src, cible, _ = items[i]
        y = VOCAB[cible]
        ids = to_ids(src).unsqueeze(0).to(DEVICE)

        modelA.train()
        logitA = modelA(ids)
        pA = torch.softmax(logitA, 1)[0]
        confA = float(pA.max().detach())
        predA = int(pA.argmax())
        maj, scoreK = mem.consult(src)
        if maj is not None and scoreK > 0.5:
            confA = min(0.99, max(confA, scoreK))
        correctA = 1.0 if predA == y else 0.0
        mem.add(src, predA, predA == y, step=i)
        optA.zero_grad()
        lossf(logitA, torch.tensor([y], device=DEVICE)).backward()
        optA.step()

        modelB.eval()
        with torch.no_grad():
            pB = torch.softmax(modelB(ids) / T, 1)[0]
        confB = float(pB.max())
        correctB = 1.0 if int(pB.argmax()) == y else 0.0

        confC = 0.5
        correctC = 1.0 if rng.randrange(NV) == y else 0.0

        for b, cf, co in (("A", confA, correctA),
                          ("B", confB, correctB),
                          ("C", confC, correctC)):
            hist[b]["conf"].append(cf)
            hist[b]["ok"].append(co)

        if (i + 1) % log_every == 0 or i == n - 1:
            for b in hist:
                hist[b]["ece"].append(ece_score(hist[b]["conf"], hist[b]["ok"]))
            print(f"[{i+1:5d}/{n}] ECE "
                  f"A={hist['A']['ece'][-1]:.3f} B={hist['B']['ece'][-1]:.3f} "
                  f"C={hist['C']['ece'][-1]:.3f}  "
                  f"accA={np.mean(hist['A']['ok']):.2f} "
                  f"accB={np.mean(hist['B']['ok']):.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    for b in hist:
        hist[b]["ece"].append(ece_score(hist[b]["conf"], hist[b]["ok"]))
    return hist


# ------------------------------------------------------------------ sorties
def sorties(hist, titres, meta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [100 * i for i in range(1, len(hist["A"]["ece"]) + 1)]
    plt.figure(figsize=(9, 5))
    for b, col in (("A", "tab:blue"), ("B", "tab:red"), ("C", "gray")):
        plt.plot(xs, hist[b]["ece"], color=col,
                 label={"A": "A bebe (web, en-ligne)",
                        "B": "B supervise (temperature)",
                        "C": "C baseline"}[b])
    plt.xlabel("interactions (issues du web)")
    plt.ylabel("ECE (10 bins)")
    plt.title("CLIENT RESEAU — ECE vs interactions (ocean controle)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "courbe_ece.png", dpi=120)
    plt.close()

    lines = ["# resultats client reseau", "",
             "meta : " + json.dumps(meta, ensure_ascii=False), "",
             "pages :"] + [f"  - {t}" for t in titres] + ["",
             "| bras | ECE final | accuracy | confiance moyenne | AURC |",
             "|---|---|---|---|---|"]
    final = {}
    for b in ("A", "B", "C"):
        confs = np.array(hist[b]["conf"])
        oks = np.array(hist[b]["ok"])
        _, _, _, aurc = risk_coverage(confs, oks)
        final[b] = {"ece_final": ece_score(confs, oks),
                    "accuracy": float(oks.mean()),
                    "conf_moy": float(confs.mean()),
                    "aurc": aurc}
        lines.append(f"| {b} | {final[b]['ece_final']:.3f} "
                     f"| {final[b]['accuracy']:.3f} "
                     f"| {final[b]['conf_moy']:.3f} "
                     f"| {final[b]['aurc']:.3f} |")
    verdict = ("ECE du bebe qui DESCEND vers B sur du vrai web = "
               "calibration qui tient hors banc."
               if final["A"]["ece_final"] < hist["A"]["ece"][0]
               else "ECE de A ne descend pas sur le web : ecrit tel quel.")
    lines += ["", "VERDICT : " + verdict]
    (OUT / "resultats_reseau.md").write_text("\n".join(lines),
                                             encoding="utf-8")
    print(json.dumps(final, indent=2))
    print("VERDICT :", verdict)
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=30)
    ap.add_argument("--items", type=int, default=200,
                    help="items web max par page")
    ap.add_argument("--rnd", type=float, default=0.15,
                    help="fraction d'items sans verite (controle honnete)")
    ap.add_argument("--delay", type=float, default=1.5,
                    help="secondes entre deux requetes")
    ap.add_argument("--interactions", type=int, default=0,
                    help="plafond sur le total (0 = tout)")
    ap.add_argument("--epochs-b", type=int, default=6)
    ap.add_argument("--local", action="store_true",
                    help="aucun appel reseau, cache seulement")
    a = ap.parse_args()

    fichiers = ensure_cache(a.pages, a.delay, a.local)
    rng = random.Random(SEED)
    items, titres = items_depuis_pages(fichiers, a.items, a.rnd, rng)
    if a.interactions:
        items = items[:a.interactions]
    if len(items) < 50:
        sys.exit(f"trop peu d'items ({len(items)}) — augmenter --items/"
                 "--pages ou verifier le texte des pages.")
    n_web = sum(1 for it in items if it[2] == "web")
    print(f"device={DEVICE}  pages={len(fichiers)}  items={len(items)} "
          f"(dont {n_web} web)  seed={SEED}", flush=True)

    modelB, T = entrainer_B(items, epochs=a.epochs_b)
    hist = mesurer(items, modelB, T)
    meta = {"pages": len(fichiers), "items": len(items),
            "web": n_web, "rnd": a.rnd, "seed": SEED,
            "mode": "local" if a.local else "reseau"}
    sorties(hist, titres, meta)
    print("sorties :", OUT)


if __name__ == "__main__":
    main()
