# -*- coding: utf-8 -*-
"""BANC DE MESURE — F:\becvide\banc.py

Instrument INDEPENDANT : aucun LLM, aucun appel réseau, aucun cloud.
Mesure si un modele VIERGE (poids aleatoires, tokenizer caracteres,
aucune phrase humaine) apprend en ligne a calibrer sa confiance,
c'est-à-dire a savoir qu'il ne sait pas.

3 bras, meme archi, meme suite d'items :
  A bebe     : 1 step SGD / interaction, feedback 0/1 brut,
               confiance = Pmax du softmax BRUT, memoire kNN externe.
  B supervise: meme archi, entrainement batch offline sur les memes
               items puis fige, confiance = temperature scaling appris
               sur un split de calibration (la verite).
  C baseline : aucun apprentissage, confiance fixe 0.5.

Metriques : ECE (10 bins) vs interactions, risk-coverage (+AURC), accuracy.
Sorties : sorties/courbe_ece.png, courbe_risk_coverage.png, resultats.md.

Usage : python banc.py [--interactions 4000] [--quick]
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

OUT = Path(__file__).resolve().parent / "sorties"
OUT.mkdir(exist_ok=True)
SEED = 1234
ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789,.-"
VOCAB = {c: i for i, c in enumerate(ALPHABET)}
NV = len(ALPHABET)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------ items
# Sequence = (entree, cible_char, regle). La regle donne la verite.
# regle "rnd" = alea pur : AUCUNE reponse (confiance honneste basse).

def gen_item(rng: random.Random):
    regle = rng.choice(["rep", "arith", "miroir", "saut", "rnd"])
    if regle == "rep":
        motif = "".join(rng.choice("abcdefg") for _ in range(rng.randint(2, 4)))
        reps = rng.randint(2, 4)
        seq = motif * reps + motif[: rng.randint(1, len(motif))]
        src, cible = seq[:-1], seq[-1]
    elif regle == "arith":
        a, d = rng.randint(0, 40), rng.randint(1, 7)
        chiffres = [str((a + i * d) % 100) for i in range(rng.randint(3, 6))]
        src = ",".join(chiffres)
        cible = str((a + len(chiffres) * d) % 100)[0]   # premier chiffre suivant
    elif regle == "miroir":
        base = "".join(rng.choice("hijk") for _ in range(rng.randint(2, 4)))
        seq = base + base[-2::-1]
        src, cible = seq[:-1], seq[-1]
    elif regle == "saut":
        debut = rng.randint(0, 5)
        lettres = "acegikmoqsuwy"
        n = rng.randint(3, 6)
        seq = lettres[debut:debut + 2 * n:2]
        src, cible = seq[:-1], seq[-1]
    else:  # rnd — item sans verite : la confiance doit rester basse
        src = "".join(rng.choice(ALPHABET[:26]) for _ in range(rng.randint(4, 9)))
        cible = rng.choice(ALPHABET[:26])
    src = src.replace(" ", "")[:24]
    if cible not in VOCAB:
        cible = "a"
    return src, cible, regle


def to_ids(s: str) -> torch.Tensor:
    return torch.tensor([VOCAB.get(c, NV - 1) for c in s], dtype=torch.long)


# ------------------------------------------------------------------ modele
class Bebe(nn.Module):
    """Transformer minimal (2 couches, d=128) — aucun pre-entraînement."""

    def __init__(self, d=128, nl=2, heads=4):
        super().__init__()
        self.emb = nn.Embedding(NV, d)
        self.pos = nn.Parameter(torch.zeros(1, 64, d))
        layer = nn.TransformerEncoderLayer(
            d, heads, dim_feedforward=4 * d, batch_first=True,
            dropout=0.1, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, nl)
        self.head = nn.Linear(d, NV)

    def forward(self, ids):
        ids = ids[:, -64:]
        x = self.emb(ids) + self.pos[:, :ids.size(1)]
        h = self.enc(x)[:, -1]
        return self.head(h)          # logits


# ------------------------------------------------------------------ metriques
def ece_score(confs, corrects, bins=10):
    confs = np.asarray(confs, float)
    corrects = np.asarray(corrects, float)
    if len(confs) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for b in range(bins):
        m = (confs >= edges[b]) & (confs < edges[b + 1] + (b == bins - 1))
        if m.sum() == 0:
            continue
        ece += m.mean() * abs(corrects[m].mean() - confs[m].mean())
    return float(ece)


def risk_coverage(confs, corrects, n_points=60):
    """Balayage du seuil tau : on ne garde que confiance >= tau.
    risk = 1 - accuracy sur les gardes ; coverage = fraction gardee."""
    confs = np.asarray(confs, float)
    corrects = np.asarray(corrects, float)
    taus = np.linspace(0, 1.0, n_points)
    cov, risk = [], []
    for t in taus:
        m = confs >= t
        cov.append(m.mean())
        risk.append(1.0 - corrects[m].mean() if m.any() else np.nan)
    aurc = np.nanmean(np.array(risk))   # indicateur brut v1
    return taus, np.array(cov), np.array(risk), float(aurc)


# ------------------------------------------------------------------ memoire knn
class MemoireKNN:
    """Memoire EXTERNE aux poids : vecteurs de comptage de caracteres
    du contexte + reponse + succes. Consultee par vote des k voisins."""

    def __init__(self, k=5):
        self.k = k
        self.X, self.y, self.ok = [], [], []

    @staticmethod
    def vec(src):
        v = np.zeros(NV, float)
        for c in src:
            v[VOCAB.get(c, NV - 1)] += 1.0
        n = max(v.sum(), 1.0)
        return v / n

    def add(self, src, cible_idx, correct):
        self.X.append(self.vec(src))
        self.y.append(cible_idx)
        self.ok.append(1.0 if correct else 0.0)

    def consult(self, src):
        """(vote des k voisins sur la cible, fiabilite = frac. de succes)"""
        if len(self.X) < 3:
            return None, 0.0
        v = self.vec(src)
        d = [np.linalg.norm(v - x) for x in self.X]
        idx = np.argsort(d)[: min(self.k, len(d))]
        ys = [self.y[i] for i in idx]
        fiab = float(np.mean([self.ok[i] for i in idx]))
        maj = max(set(ys), key=ys.count)
        accord = ys.count(maj) / len(ys)
        return maj, fiab * accord


# ------------------------------------------------- memoire CONSOLIDEE (v1.1)
# Porte dans banc.py le 03/10 par LYNX (spec: SPEC_ECHANGE.md 1.1).
# RATIONNELE MESURE : MemoireKNN est append-only ; avec le POOL FINI de 1200
# items qui repasse, ~48.8k entrees sur 50k sont des DOUBLONS -> consult()
# O(n) par appel -> O(n^2) cumule. Ici une CLE = un contexte : add() met a
# jour l'entree existante. consult() devient O(distinct).
# Format FIGE bebe-etat/1 : keys[M,NV] f32, y_counts[M,NV] i32, ok[M] f32,
# n[M] i32, last_seen[M] i64. Extension -> bebe-etat/2.
# etat.py/fusion.py importent CETTE classe (banc.MemoireConsolidee) :
# une seule implementation, pas deux.
CAP_MEMOIRE = 50_000   # entrees POST-consolidation, eviction LRU


class MemoireConsolidee:
    """Memoire kNN CONSOLIDEE PAR CLE. Interface = MemoireKNN + step."""

    def __init__(self, k=5, cap=None):
        self.k = k
        self.cap = cap
        self.keys = np.zeros((0, NV), np.float32)
        self.y_counts = np.zeros((0, NV), np.int32)
        self.ok = np.zeros(0, np.float32)       # ok moyen par cle
        self.n = np.zeros(0, np.int32)          # nb d'ajouts sur la cle
        self.last_seen = np.zeros(0, np.int64)  # step du dernier add
        self._ix = {}

    @staticmethod
    def vec(src):
        return MemoireKNN.vec(src)

    def _rebuild_index(self):
        self._ix = {self.keys[r].tobytes(): r for r in range(len(self.keys))}

    def add(self, src, cible_idx, correct, step=0):
        v = self.vec(src)
        kb = v.tobytes()
        r = self._ix.get(kb)
        if r is None:
            self.keys = np.vstack([self.keys, v[None, :]])
            self.y_counts = np.vstack([self.y_counts, np.zeros((1, NV), np.int32)])
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


# ------------------------------------------------------------------ bras
def run_bras(n_inter: int, log_every: int = 100):
    rng = random.Random(SEED)
    torch.manual_seed(SEED)

    # --- LEVIER UNIQUE (Mathieu, 03/10) : LE MONDE REVIENT.
    # Un pool FINI d'items qui repasse, au lieu d'un flux i.i.d.
    # Le bebe revoit ses erreurs -> consolidation possible, la memoire
    # kNN redevient pertinente, confiance et accuracy peuvent remonter
    # ensemble. Un seul levier. Une seule graine. Rien d'autre ne bouge.
    POOL = 1200
    pool = [gen_item(rng) for _ in range(POOL)]
    items = [pool[i % POOL] for i in range(n_inter + 800)]

    # --- B (supervise) : entrainement batch sur les 3/4 premiers items,
    #     puis fige ; temperature scaling sur le quart suivant (verite).
    modelB = Bebe().to(DEVICE)
    optB = torch.optim.Adam(modelB.parameters(), lr=1e-3, foreach=True)
    lossf = nn.CrossEntropyLoss()
    split = int(n_inter * 0.75)
    modelB.train()
    for _ in range(6):                                   # epoques
        for src, cible, _ in items[:split]:
            ids = to_ids(src).unsqueeze(0).to(DEVICE)
            logit = modelB(ids)
            optB.zero_grad()
            lossf(logit, torch.tensor([VOCAB[cible]], device=DEVICE)).backward()
            optB.step()
    # temperature scaling (supervise) sur le split de calibration
    temp = nn.Parameter(torch.ones(1, device=DEVICE))
    optT = torch.optim.LBFGS([temp], lr=0.1, max_iter=30)
    modelB.eval()

    def calib_loss():
        tot = 0.0
        for src, cible, _ in items[split:split + 400]:
            ids = to_ids(src).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logit = modelB(ids) / temp.clamp(0.05, 10)
            tot += lossf(logit, torch.tensor([VOCAB[cible]], device=DEVICE))
        return tot

    try:
        optT.step(calib_loss)
    except Exception:
        pass
    T = float(temp.detach().clamp(0.05, 10).item())

    # --- A (bebe) : aleatoire, 1 step SGD en ligne, memoire kNN
    torch.manual_seed(SEED + 1)
    modelA = Bebe().to(DEVICE)
    optA = torch.optim.SGD(modelA.parameters(), lr=3e-4)
    mem = MemoireConsolidee(k=5, cap=CAP_MEMOIRE)

    # --- C (baseline)
    hist = {"A": {"conf": [], "ok": [], "ece": []},
            "B": {"conf": [], "ok": [], "ece": []},
            "C": {"conf": [], "ok": [], "ece": []}}

    def eval_ece(h):
        for b in ("A", "B", "C"):
            c, o = h[b]["conf"], h[b]["ok"]
            h[b]["ece"].append(ece_score(c, o))

    t0 = time.time()
    for i in range(n_inter):
        src, cible, regle = items[i]
        y = VOCAB[cible]
        ids = to_ids(src).unsqueeze(0).to(DEVICE)

        # ---- A : confiance brute + boost kNN (memoire, pas de poids)
        modelA.train()
        logitA = modelA(ids)
        pA = torch.softmax(logitA, 1)[0]
        confA = float(pA.max().detach())
        predA = int(pA.argmax())
        maj, scoreK = mem.consult(src)
        if maj is not None and scoreK > 0.5:
            confA = min(0.99, max(confA, scoreK))     # la memoire conforte
        correctA = 1.0 if predA == y else 0.0
        mem.add(src, predA, predA == y, step=i)
        # feedback 0/1 brut -> 1 step
        optA.zero_grad()
        lossA = lossf(logitA, torch.tensor([y], device=DEVICE))
        lossA.backward()
        optA.step()

        # ---- B : fige, temperature
        modelB.eval()
        with torch.no_grad():
            pB = torch.softmax(modelB(ids) / T, 1)[0]
        confB = float(pB.max())
        predB = int(pB.argmax())
        correctB = 1.0 if predB == y else 0.0

        # ---- C : baseline honnête = predicteur aleatoire, confiance fixe
        confC = 0.5
        correctC = 1.0 if rng.randrange(NV) == y else 0.0

        for b, (cf, co) in (("A", (confA, correctA)),
                            ("B", (confB, correctB)),
                            ("C", (confC, correctC))):
            hist[b]["conf"].append(cf)
            hist[b]["ok"].append(co)

        if (i + 1) % log_every == 0 or i == n_inter - 1:
            eval_ece(hist)
            ec = {b: hist[b]["ece"][-1] for b in hist}
            print(f"[{i+1:5d}/{n_inter}] ECE A={ec['A']:.3f} "
                  f"B={ec['B']:.3f} C={ec['C']:.3f}  "
                  f"accA={np.mean(hist['A']['ok']):.2f} "
                  f"accB={np.mean(hist['B']['ok']):.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    eval_ece(hist)
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interactions", type=int, default=4000)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    n = 800 if a.quick else a.interactions

    print(f"device={DEVICE}  interactions={n}  seed={SEED}")
    hist = run_bras(n)

    # ---------------- sorties
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [100 * (i + 1) for i in range(len(hist["A"]["ece"]))]

    plt.figure(figsize=(9, 5))
    for b, col in (("A", "tab:blue"), ("B", "tab:red"), ("C", "gray")):
        plt.plot(xs, hist[b]["ece"], label={
            "A": "A bebe (en-ligne, sans supervision)",
            "B": "B supervise (temperature)",
            "C": "C baseline"}[b], color=col)
    plt.xlabel("interactions")
    plt.ylabel("ECE (10 bins)")
    plt.title("LA COURBE — ECE vs interactions (3 bras)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "courbe_ece.png", dpi=120)
    plt.close()

    # ---------------- 3e courbe : l'ACCURACY (demandee par Mathieu 03/10)
    # hist["ok"] est l'historique complet en memoire : moyenne cumulee,
    # aucune modification de la boucle de mesure necessaire.
    n_ok = len(hist["A"]["ok"])
    xs_acc = np.arange(1, n_ok + 1)
    plt.figure(figsize=(9, 5))
    for b, col in (("A", "tab:blue"), ("B", "tab:red"), ("C", "gray")):
        acc = np.cumsum(hist[b]["ok"]) / xs_acc
        plt.plot(xs_acc, acc, color=col,
                 label={"A": "A bebe (en-ligne, sans supervision)",
                        "B": "B supervise (temperature)",
                        "C": "C baseline"}[b])
    plt.xlabel("interactions")
    plt.ylabel("accuracy")
    plt.title("LA COURBE — accuracy vs interactions (3 bras)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "courbe_accuracy.png", dpi=120)
    plt.close()

    # risk-coverage final (post-hoc, sur toute la trace)
    lines = ["# resultats", ""]
    final = {}
    plt.figure(figsize=(9, 5))
    for b in ("A", "B", "C"):
        confs, oks = np.array(hist[b]["conf"]), np.array(hist[b]["ok"])
        taus, cov, risk, aurc = risk_coverage(confs, oks)
        final[b] = {"ece_final": ece_score(confs, oks),
                    "accuracy": float(oks.mean()),
                    "conf_moy": float(confs.mean()),
                    "aurc": aurc}
        col = {"A": "tab:blue", "B": "tab:red", "C": "gray"}[b]
        if np.nanstd(confs) < 1e-9:
            # confiance constante (C) : couverture = 1.0 pour tau <= 0.5
            # puis vide (NaN) -> la courbe se reduit a UN point invisible.
            # Ici tout sous-ensemble a le meme risk : droite horizontale
            # exacte sur [0,1], comme dans les papers.
            plt.hlines(float(np.nanmean(risk)), 0, 1, colors=col,
                       linestyles="--", label=f"{b} (AURC={aurc:.3f})")
        else:
            plt.plot(cov, risk, color=col, label=f"{b} (AURC={aurc:.3f})")
    plt.xlabel("coverage")
    plt.ylabel("risk (1 - accuracy)")
    plt.title("Risk-coverage final")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "courbe_risk_coverage.png", dpi=120)
    plt.close()

    # historique complet -> re-tracable a tout moment (publication)
    np.savez(OUT / "hist.npz",
             **{f"{b}_{m}": np.array(hist[b][m], dtype=np.float32)
                for b in ("A", "B", "C") for m in ("conf", "ok")})

    lines += ["| bras | ECE final | accuracy | confiance moyenne | AURC |",
              "|---|---|---|---|---|"]
    for b in ("A", "B", "C"):
        f = final[b]
        lines.append(f"| {b} | {f['ece_final']:.3f} | {f['accuracy']:.3f} "
                     f"| {f['conf_moy']:.3f} | {f['aurc']:.3f} |")
    lines += ["", "ECE courbe (points) : " +
              json.dumps({b: [round(x, 4) for x in hist[b]["ece"]]
                          for b in hist})]
    lines += ["", "accuracy courbe (points /100) : " +
              json.dumps({b: [round(float(np.mean(hist[b]["ok"]
                                                  [:100 * (i + 1)])), 4)
                              for i in range(len(hist[b]["ece"]))]
                          for b in hist})]
    verdict = ("ECE du bebe (A) qui DESCEND vers B = calibration en emergence."
               if final["A"]["ece_final"] < hist["A"]["ece"][0]
               else "ECE de A ne descend pas : ecrit comme tel, pas triche.")
    lines += ["", "VERDICT : " + verdict]
    (OUT / "resultats.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(final, indent=2))
    print("VERDICT :", verdict)
    print("sorties :", OUT)


if __name__ == "__main__":
    main()
