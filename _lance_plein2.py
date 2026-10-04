# -*- coding: utf-8 -*-
"""PLEIN Phase 2 — PRIMARY compute-matched (protocole §7 + §7bis contre-signé a762670).

3 niches x S=10 graines x 2 bras, budget 600 s, VAGUES DE 6 processus
(parallel 6 = 12 threads logiques, machine 12 logiques/6 physiques — validated pilote).

Graines §7bis (figees, inchangees) :
  n0 rep          data 17..26 / init 2017..2026  (17/18/19 = pilote -> REUTILISES, pas relances)
  n1 arith,miroir data 27..36 / init 2027..2036
  n2 saut,rnd     data 37..46 / init 2037..2046
Nouveaux runs : 14 (n0) + 20 (n1) + 20 (n2) = 54 -> 9 vagues ~ 100 min.

Sorties : resultats/p2_plein/<nich>_s<sd>_<bras>/ (etat/ + run.log)
          resultats/p2_plein/runs_plein.json  {"n0": {"17": {"nu": dir, "miel": dir}, ...}, ...}
             (n0 17/18/19 pointent vers resultats/p2_pilote/ = reuse)
          resultats/p2_plein/plein_meta.json  telemetrie par run + codes + vagues
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

R = Path(__file__).resolve().parent
OUT = R / "resultats" / "p2_plein"
PILOT = R / "resultats" / "p2_pilote"
MIEL = r"F:\becvide\miel_phase1.npz"
NICHES = [
    ("n0", "rep",          [(17 + k, 2017 + k) for k in range(10)]),
    ("n1", "arith,miroir", [(27 + k, 2027 + k) for k in range(10)]),
    ("n2", "saut,rnd",     [(37 + k, 2037 + k) for k in range(10)]),
]
REUSE = {("n0", 17), ("n0", 18), ("n0", 19)}  # deja pousses (p2_pilote)
BUDGET = "600"
WAVE = 6

env = dict(os.environ)
env.update({"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2", "NUMEXPR_NUM_THREADS": "2"})

# file d'attente : (nom, regles, sd, si, bras)
queue = []
for nic, regles, seeds in NICHES:
    for sd, si in seeds:
        if (nic, sd) in REUSE:
            continue
        for bras in ("nu", "miel"):
            queue.append((f"{nic}_s{sd}_{bras}", regles, sd, si, bras, nic))

print(f"{len(queue)} runs neufs, vagues de {WAVE}", flush=True)

codes = {}
t_start = time.time()
for i in range(0, len(queue), WAVE):
    batch = queue[i:i + WAVE]
    n_vague = i // WAVE + 1
    procs = {}
    for nom, regles, sd, si, bras, nic in batch:
        out = OUT / nom
        out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(R / "scripts" / "entrainer_local.py"),
               "--n", "100000",
               "--seed-data", str(sd),
               "--regles", regles,
               "--budget-seconds", BUDGET,
               "--eval-every", "500",
               "--seed", str(si),
               "--out", str(out)]
        if bras == "miel":
            cmd += ["--miel", MIEL]
        log = open(out / "run.log", "w", encoding="utf-8")
        p = subprocess.Popen(cmd, cwd=str(R), env=env, stdout=log,
                             stderr=subprocess.STDOUT)
        procs[nom] = (p, log)
    print(f"vague {n_vague}/{(len(queue) + WAVE - 1) // WAVE} : "
          + ", ".join(n for n, *_ in batch), flush=True)
    for nom, (p, log) in procs.items():
        rc = p.wait()
        log.close()
        codes[nom] = rc
    print(f"vague {n_vague} terminee rc={sorted(set(codes[n] for n, *_ in batch))} "
          f"t+{int(time.time() - t_start)}s", flush=True)

ok = all(rc == 0 for rc in codes.values())

# runs_plein.json — schema niche -> seed -> bras ; reuse = dirs p2_pilote
runs = {}
for nic, _regles, seeds in NICHES:
    runs[nic] = {}
    for sd, _si in seeds:
        if (nic, sd) in REUSE:
            runs[nic][str(sd)] = {b: str(PILOT / f"s{sd}_{b}") for b in ("nu", "miel")}
        else:
            runs[nic][str(sd)] = {b: str(OUT / f"{nic}_s{sd}_{b}") for b in ("nu", "miel")}
(OUT / "runs_plein.json").write_text(json.dumps(runs, indent=1), encoding="utf-8")

# telemetrie : les 54 neufs + indicateur sur les 6 reutilises
telem = {"codes": codes, "reused": sorted(f"{n}{s}" for n, s in REUSE),
         "vagues": (len(queue) + WAVE - 1) // WAVE,
         "duree_s": int(time.time() - t_start)}
for nom in codes:
    try:
        m = json.loads((OUT / nom / "etat" / "meta.json").read_text(encoding="utf-8"))
        telem[nom] = {"n_interactions": m.get("n_interactions"),
                      "heldout_sha256": m.get("heldout_sha256"),
                      "code_sha": m.get("code_sha"),
                      "regles": m.get("regles"),
                      "miel_taux_activation": (m.get("miel") or {}).get("taux_activation")}
    except Exception as e:
        telem[nom] = {"erreur": str(e)}
(OUT / "plein_meta.json").write_text(json.dumps(telem, indent=1), encoding="utf-8")

print(json.dumps({k: telem[k] for k in ("codes", "reused", "vagues", "duree_s")},
                 indent=1), flush=True)
print("PLEIN PRIMARY OK" if ok else f"ECHEC : {codes}", flush=True)
sys.exit(0 if ok else 1)
