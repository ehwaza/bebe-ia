# -*- coding: utf-8 -*-
"""bebe/nectar.py — format de la LIGNE DE NECTAR (abeille -> miel).

Spec v1 figee le 04/10 (LYNX + Claude, mailbox) :
  {"v":1,"abeille":"hex16","niche":"regles=...; data_seed=N","t":ISO8601,
   "genome":{"heldout_sha256","code_sha","arch"},
   "seen":{"n_items","cycles"},
   "nectar":{"k","m","m_total",
             "keys","y_counts","n","ok_num","last_seen"},   <- b64 raw
   "h":sha256(payload sans h, sort_keys, separators compacts)}

Principe : une ligne = UNE argument de merge_memoires() (union par cle +
somme des compteurs). Le miel = fold de merge_memoires() sur les lignes —
cote AGREGATEUR (moitie Claude), pas ici.

Invariants techniques :
  - ok_num = SOMME de ok*n (jamais ok) : le fold re-divise par n — on
    preserve l'associativite de la somme (reconstruction ok = ok_num/n).
  - y_counts reshape (m, -1) : le nombre de classes c n'est PAS un champ
    supplementaire (spec figee) — il est derivable de la taille.
  - h = integrite basique v0 (pas une signature : l'attaquant peut re-hasher).
    HMAC par relais = v1.
  - Duck-typed : n'importe quel objet portant les 5 tableaux (etat.Memoire-
    Consolidee OU banc.MemoireConsolidee — les deux existent dans le repo ;
    le writer ne se lie a aucune des deux).
"""
import base64
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
try:
    import banc  # noqa: E402
    CAP = int(banc.CAP_MEMOIRE)
except Exception:  # banc/torch indisponible -> validation seule
    CAP = 50_000

VERSION = 1
NICHE_MIN = "regles="  # motif minimal d'une niche valide
# abeille = 16 hex minuscules (64 bits opaques, non identifiant — spec v1)
ABEILLE_RE = re.compile(r"^[0-9a-f]{16}$")


# ------------------------------------------------------------------ b64/utils
def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def _raw(b64: str, nbytes: int) -> bytes:
    buf = base64.b64decode(b64, validate=True)
    if len(buf) != nbytes:
        raise ValueError(f"b64: {len(buf)} octets, attendu {nbytes}")
    return buf


def _canon(ligne: dict) -> str:
    """Payload canonique pour le hash : sans h, cles triees, compact."""
    d = {k: v for k, v in ligne.items() if k != "h"}
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


def _h(ligne: dict) -> str:
    return hashlib.sha256(_canon(ligne).encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ ecriture
def ligne_nectar(*, abeille: str, niche: str, meta: dict, seen: dict,
                 mem, m_total=None) -> dict:
    """Construit (et hache) la ligne a partir d'une memoire consolidatee.

    meta   : meta.json du bundle de l'abeille (genome subset extrait ici).
    seen   : {"n_items": int, "cycles": int} — provenance des comptes.
    m_total: nb d'entrees AVANT prune (transparence ; defaut = m).
    """
    k = np.asarray(mem.keys, np.float32)
    yc = np.asarray(mem.y_counts, np.int32)
    n = np.asarray(mem.n, np.int32)
    ok = np.asarray(mem.ok, np.float32)
    ls = np.asarray(mem.last_seen, np.int64)
    if k.ndim != 2:
        raise ValueError(f"keys doit etre (m,k), obtenu {k.shape}")
    m = int(k.shape[0])
    for name, a in (("y_counts", yc), ("n", n), ("ok", ok), ("last_seen", ls)):
        if len(a) != m:
            raise ValueError(f"{name}: {len(a)} entrees != m={m}")
    if m and yc.size % m:
        raise ValueError(f"y_counts: {yc.size} non divisible par m={m}")

    # ok_num = somme de ok*n  (associativite : le fold re-divise par n)
    ok_num = (ok.astype(np.float64) * n.astype(np.float64)).astype(np.float32)

    ligne = {
        "v": VERSION,
        "abeille": str(abeille),
        "niche": str(niche),
        "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "genome": {
            "heldout_sha256": meta.get("heldout_sha256"),
            "code_sha": meta.get("code_sha"),
            "arch": meta.get("arch"),
        },
        "seen": {"n_items": int(seen.get("n_items", 0)),
                 "cycles": int(seen.get("cycles", 0))},
        "nectar": {
            "k": int(k.shape[1]),
            "m": m,
            "m_total": int(m_total) if m_total is not None else m,
            "keys": _b64(k),
            "y_counts": _b64(yc),
            "n": _b64(n),
            "ok_num": _b64(ok_num),
            "last_seen": _b64(ls),
        },
    }
    ligne["h"] = _h(ligne)
    return ligne


# ------------------------------------------------------------------ lecture
def lire_nectar(ligne: dict):
    """Decode APRES validation. Retourne (keys, y_counts, n, ok_num, last_seen).
    NE FUSE PAS : le fold merge_memoires() appartient a l'agregateur."""
    nex = ligne["nectar"]
    m, k = int(nex["m"]), int(nex["k"])
    keys = np.frombuffer(_raw(nex["keys"], m * k * 4),
                         dtype=np.float32).reshape(m, k).copy()
    yc_buf = base64.b64decode(nex["y_counts"], validate=True)
    c = (len(yc_buf) // 4) // m if m else 0
    yc = (np.frombuffer(yc_buf, dtype=np.int32).reshape(m, c).copy()
          if m else np.zeros((0, 0), np.int32))
    n = np.frombuffer(_raw(nex["n"], m * 4), dtype=np.int32).copy()
    ok_num = np.frombuffer(_raw(nex["ok_num"], m * 4),
                           dtype=np.float32).copy()
    ls = np.frombuffer(_raw(nex["last_seen"], m * 8),
                       dtype=np.int64).copy()
    return keys, yc, n, ok_num, ls


# ------------------------------------------------------------------ validation
def valider(ligne) -> tuple:
    """(ok, raisons) — le relais refuse TOUTE ligne non conforme AVANT fold."""
    r = []

    def dur(cond, msg):
        if not cond:
            r.append(msg)

    dur(isinstance(ligne, dict), "pas un dict")
    if r:
        return False, r
    dur(ligne.get("v") == VERSION, f"version != {VERSION}")
    ab = ligne.get("abeille")
    dur(isinstance(ab, str) and ABEILLE_RE.match(ab) is not None,
        "abeille != hex16 (^[0-9a-f]{16}$)")
    ni = ligne.get("niche")
    dur(isinstance(ni, str) and NICHE_MIN in ni and "data_seed" in ni,
        "niche mal formee (attendu 'regles=...; data_seed=N')")
    g = ligne.get("genome") or {}
    dur(bool(g.get("heldout_sha256")) and bool(g.get("code_sha"))
        and g.get("arch") is not None,
        "genome incomplet (heldout_sha256/code_sha/arch)")
    s = ligne.get("seen") or {}
    dur(isinstance(s.get("n_items"), int) and isinstance(s.get("cycles"), int),
        "seen invalide")
    # intégrité : hash du payload canonique
    dur(isinstance(ligne.get("h"), str) and ligne.get("h") == _h(ligne),
        "hash invalide (alteration)")

    nex = ligne.get("nectar") or {}
    try:
        m, k = int(nex["m"]), int(nex["k"])
        dur(m >= 0 and k >= 0, "m/k negatifs")
        dur(m <= CAP, f"m={m} > CAP_MEMOIRE={CAP}")
        dur(int(nex["m_total"]) >= m, "m_total < m (prune incoherent)")
        # decodages b64 avec tailles exactes (levent ValueError)
        _raw(nex["keys"], m * k * 4)
        _raw(nex["ok_num"], m * 4)
        _raw(nex["last_seen"], m * 8)
        yc_buf = base64.b64decode(nex["y_counts"], validate=True)
        dur(len(yc_buf) % 4 == 0, "y_counts: taille non multiple de 4")
        if len(yc_buf) % 4 == 0:
            dur(m == 0 or (len(yc_buf) // 4) % m == 0,
                "y_counts non divisible par m")
        n = np.frombuffer(_raw(nex["n"], m * 4), dtype=np.int32)
        ok_num = np.frombuffer(_raw(nex["ok_num"], m * 4), dtype=np.float32)
        dur(bool((n >= 0).all()), "n negatif")
        dur(bool((ok_num >= -1e-6).all()), "ok_num negatif")
        dur(bool((ok_num <= n * (1 + 1e-4) + 1e-4).all()),
            "ok_num > n (compteur incoherent : ok<=1 impose)")
    except (KeyError, TypeError, ValueError) as e:
        r.append(f"nectar illisible: {e}")

    return (len(r) == 0), (r if r else ["OK"])


# ------------------------------------------------------------------ jsonl
def ecrire_jsonl(path, lignes) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for li in lignes:
            f.write(json.dumps(li, sort_keys=True,
                               separators=(",", ":")) + "\n")
    return len(lignes)


def lire_jsonl(path):
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out
