# -*- coding: utf-8 -*-
"""Sensibilite STEP-MATCHED (secondaire) — declencheur pre-ecrit Claude (04/10 05:1x) :
gap d'interactions primaire > 5 %  OU  E2-E0 <= 0  =>  eval A PAS EGAL obligatoire.
Mesure : gap = 6.09 % (> 5 % sur les 3 graines) -> CONDITION #1 SATISFAITE.

Le primaire compute-matched (p2_pilote, budget 600 s) n'est PAS toche.
Le step-matched = meme graines/bras/pools, MAIS --n 50000 EXACT pour les 2 bras
(n_fait identique a l'item pres) — budget 900 s = garde-fou purement securitaire
(le n lie, ~505-540 s reel ; aucun run ne doit atteindre le budget).

Pourquoi un re-run et pas les checkpoints : --ckpt-every 25000 n'a ECRIT AUCUN
checkpoint dans p2_pilote (seul le modele final existe) — donc l'eval a pas egal
passe par un entrainement a n fixe. Documente comme tel au rapport.

Sorties : resultats/p2_step/s<graine>_<bras>/ + runs_step.json (meme schema que runs.json)
          + step_meta.json (telemetrie : n_interactions DOIT etre 50000 partout)
"""
import json
import os
import subprocess
import sys
from pathlib import Path

R = Path(__file__).resolve().parent
OUT = R / "resultats" / "p2_step"
MIEL = r"F:\becvide\miel_phase1.npz"
GRAINES = [(17, 2017), (18, 2018), (19, 2019)]
N_STEPS = "50000"   # pas exact, commun aux 2 bras
BUDGET = "900"      # garde-fou (ne doit PAS lier : n=50000 ~ 540s max)

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
               "--n", N_STEPS,
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
        print(f"{nom} lance : pid={p.pid} n={N_STEPS} data_seed={sd} init={si} "
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
(OUT / "runs_step.json").write_text(json.dumps(runs, indent=1), encoding="utf-8")

telem = {"codes": codes, "n_cible": int(N_STEPS)}
for nom in procs:
    try:
        m = json.loads((OUT / nom / "etat" / "meta.json").read_text(encoding="utf-8"))
        telem[nom] = {"n_interactions": m.get("n_interactions"),
                      "heldout_sha256": m.get("heldout_sha256"),
                      "code_sha": m.get("code_sha"),
                      "miel_taux_activation": (m.get("miel") or {}).get("taux_activation")}
    except Exception as e:
        telem[nom] = {"erreur": str(e)}
(OUT / "step_meta.json").write_text(json.dumps(telem, indent=1), encoding="utf-8")

# garde-fou step-matched : n DOIT etre egal partout, sinon le secondaire est inutilisable
ns = [telem[n].get("n_interactions") for n in procs if "erreur" not in telem[n]]
strict = bool(ns) and len(set(ns)) == 1 and ns[0] == int(N_STEPS)
telem["step_matched_strict"] = strict
(OUT / "step_meta.json").write_text(json.dumps(telem, indent=1), encoding="utf-8")

print(json.dumps(telem, indent=1), flush=True)
print("STEP-MATCHED OK" if (ok and strict)
      else f"ECHEC/DETREINTE : ok={ok} strict={strict}", flush=True)
sys.exit(0 if (ok and strict) else 1)
