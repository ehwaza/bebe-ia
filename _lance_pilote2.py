# -*- coding: utf-8 -*-
"""Pilote Phase 2 — protocole §7 : niche REP x 3 graines x 2 bras, budget 600 s.

Graines (declarees §7) : data_seed 17/18/19, init 2017/2018/2019.
Bras : T_nu (sans --miel) et T_miel (--miel miel_phase1.npz) — SEUL le miel differe
(memes seeds, meme source/regles/defaults => meme ordre de donnees).
6 processus EN PARALLELE (condition de charge homogene, estime §7 ~12 min).

Sorties :
  resultats/p2_pilote/s<graine>_<bras>/   (etat/ + run.log, meme layout Phase 1)
  resultats/p2_pilote/runs.json           {"17": {"nu": dir, "miel": dir}, "18": ..., "19": ...}
  resultats/p2_pilote/pilot_meta.json     telemetrie d'appariement : n_interactions par run
                                          + meta.miel (taux_activation/sk) pour T_miel
"""
import json
import os
import subprocess
import sys
from pathlib import Path

R = Path(__file__).resolve().parent
OUT = R / "resultats" / "p2_pilote"
MIEL = r"F:\becvide\miel_phase1.npz"
GRAINES = [(17, 2017), (18, 2018), (19, 2019)]
BUDGET = "600"  # secondes — parite Phase 1

env = dict(os.environ)
env.update({"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2", "NUMEXPR_NUM_THREADS": "2"})

procs = {}
for sd, si in GRAINES:
    for bras in ("nu", "miel"):
        nom = f"s{sd}_{bras}"
        out = OUT / nom
        out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(R / "scripts" / "entrainer_local.py"),
               "--n", "100000",
               "--seed-data", str(sd),
               "--regles", "rep",
               "--budget-seconds", BUDGET,
               "--eval-every", "500",
               "--seed", str(si),
               "--out", str(out)]
        if bras == "miel":
            cmd += ["--miel", MIEL]
        log = open(out / "run.log", "w", encoding="utf-8")
        p = subprocess.Popen(cmd, cwd=str(R), env=env, stdout=log,
                             stderr=subprocess.STDOUT)
        procs[nom] = {"p": p, "log": log, "sd": sd, "bras": bras}
        print(f"{nom} lance : pid={p.pid} data_seed={sd} init={si} "
              f"miel={'OUI' if bras == 'miel' else 'non'}", flush=True)

codes = {}
for nom, d in procs.items():
    rc = d["p"].wait()
    d["log"].close()
    codes[nom] = rc
    print(f"{nom} termine rc={rc}", flush=True)

ok = all(rc == 0 for rc in codes.values())

runs = {str(sd): {b: str(OUT / f"s{sd}_{b}") for b in ("nu", "miel")}
        for sd, _ in GRAINES}
(OUT / "runs.json").write_text(json.dumps(runs, indent=1), encoding="utf-8")

telem = {"codes": codes}
for nom, d in procs.items():
    try:
        m = json.loads((OUT / nom / "etat" / "meta.json").read_text(encoding="utf-8"))
        telem[nom] = {"n_interactions": m.get("n_interactions"),
                      "n_new": m.get("n_new"),
                      "regles": m.get("regles"),
                      "heldout_sha256": m.get("heldout_sha256"),
                      "code_sha": m.get("code_sha"),
                      "miel": m.get("miel")}  # None sur T_nu, dict sur T_miel
    except Exception as e:  # run casse -> trace lisible
        telem[nom] = {"erreur": str(e)}
(OUT / "pilot_meta.json").write_text(json.dumps(telem, indent=1), encoding="utf-8")

print(json.dumps(telem, indent=1), flush=True)
print("PILOTE OK" if ok else f"ECHEC : {codes}", flush=True)
sys.exit(0 if ok else 1)
