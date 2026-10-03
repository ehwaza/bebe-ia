# -*- coding: utf-8 -*-
"""scripts/synchroniser.py — synchronisation local <-> fragment GitHub.

Protocole (Mathieu, 04/10) :
  1. Detecte le dernier etat dans le depot (etats/etat_courant).
  2. Le telecharge (git pull).
  3. Compare avec l'etat local (etats/local).
  4. Lance la FUSION PAR CONFIANCE si divergence.
  5. Upload l'etat fusionne (commit + push).

Regles de coordination (Triangle) : pull AVANT push, un seul push a la fois,
jamais de regeneration du held-out.

Usage :
  python scripts/synchroniser.py                      # cycle complet
  python scripts/synchroniser.py --local etats/local  # emplacement different
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bebe.fusion import fusionner_confiance  # noqa: E402


def git(*args):
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} -> {r.stderr.strip()}")
    return r.stdout


def memes_etats(m1, m2):
    """Deux etats sont-ils le meme ? (identite de filiation + comptes)"""
    return (m1.get("fragment") == m2.get("fragment")
            and m1.get("cycle") == m2.get("cycle")
            and m1.get("n_interactions") == m2.get("n_interactions"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", default=str(ROOT / "etats" / "local"))
    ap.add_argument("--remote", default=str(ROOT / "etats" / "etat_courant"))
    ap.add_argument("--heldout", default=str(ROOT / "heldout.npz"))
    ap.add_argument("--push", action="store_true", help="commit + push du resultat")
    a = ap.parse_args()

    # 1+2. dernier etat du depot
    print("== pull du depot (regle : pull avant push)")
    try:
        git("pull", "--rebase", "--autostash")
    except RuntimeError as e:
        print("pull impossible (depot vierge ?) :", e)

    local, remote = Path(a.local), Path(a.remote)
    dossier_etat = lambda d: d if (d / "meta.json").exists() else d / "etat"  # noqa: E731

    if not (dossier_etat(remote) / "meta.json").exists():
        print("pas d'etat distant : rien a synchroniser")
        return
    if not (dossier_etat(local) / "meta.json").exists():
        print("pas d'etat local : premier telechargement")
        import shutil
        local.mkdir(parents=True, exist_ok=True)
        for f in dossier_etat(remote).iterdir():
            shutil.copy(f, dossier_etat(local))
        return

    m_loc = json.loads((dossier_etat(local) / "meta.json").read_text(encoding="utf-8"))
    m_rem = json.loads((dossier_etat(remote) / "meta.json").read_text(encoding="utf-8"))

    # 3. comparaison
    if memes_etats(m_loc, m_rem):
        print("etats identiques : rien a faire")
        return

    # 4. fusion par confiance si divergence
    print(f"== divergence : local {m_loc.get('fragment')}#c{m_loc.get('cycle')} "
          f"(n={m_loc.get('n_interactions')}) vs distant "
          f"{m_rem.get('fragment')}#c{m_rem.get('cycle')} (n={m_rem.get('n_interactions')})")
    tmp = ROOT / "etats" / "fusionnee"
    r = fusionner_confiance([str(dossier_etat(local)), str(dossier_etat(remote))],
                            a.heldout, str(tmp / "etat"))
    if not r.get("ok"):
        print("FUSION REFUSEE :")
        for rais in r["raisons"]:
            print("  -", rais)
        sys.exit(1)
    print("fusion OK : poids =", [round(w, 3) for w in r["poids"]],
          "| ECE fragments =", [round(e, 4) for e in r["eces_fragments"]])

    # 5. upload de l'etat fusionne
    import shutil
    dst = dossier_etat(remote)
    for f in (tmp / "etat").iterdir():
        shutil.copy(f, dst)
    if a.push:
        git("add", "etats")
        st = git("status", "--porcelain", "etats")
        if st.strip():
            git("commit", "-m", f"synchronise : fusion par confiance "
                                f"(c{r.get('eces_fragments') and 'ok'})")
            git("push")
            print("push effectue")
        else:
            print("rien a commiter")
    else:
        print("etat fusionne pose dans", dst, "(--push pour uploader)")


if __name__ == "__main__":
    main()
