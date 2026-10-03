# bebe-ia — le bebe apprend a savoir qu'il ne sait pas

Un modele vierge (poids aleatoires, aucun pre-entrainement, aucun LLM dans la
boucle) apprend **en ligne, sans supervision de confiance** : il ne voit que
juste/faux, et sa confiance (Pmax du softmax brut, booste par une memoire kNN
externe) doit apprendre seule a refleter sa competence reelle.

Mesure : **ECE** (calibration) vs interactions, **risk-coverage / AURC**,
accuracy. Le held-out (`heldout.npz`) est **FIGE** — jamais regenere.

## Structure

```
bebe-ia/
├── banc.py              instrument de mesure 3 bras (SOURCE UNIQUE du genome)
├── client_reseau.py     client de fragment reseau
├── etat.py              format de paquet bebe-etat/1 (C1-C5)
├── bebe/                package : modele.py, metriques.py, fusion.py
├── scripts/             entrainer_local, entrainer_github, synchroniser,
│                        fusionner, multi_seeds, analyse
├── etats/               checkpoints (etat_courant = fragment GitHub, local)
├── resultats/           courbes, logs, rapports
└── .github/workflows/apprendre.yml   fragment GitHub (cron 6 h, budget 300 min)
```

**Non-duplication** : `bebe/modele.py` et `bebe/metriques.py` re-exportent
banc.py. Le genome existe une seule fois.

## Le triangle d'apprentissage

- **fragment local** : `python scripts/entrainer_local.py --n 100000 --out resultats/local`
- **fragment GitHub** : `.github/workflows/apprendre.yml` (budget temps, plafond
  5 M interactions, debit mesure dans `etats/etat_courant/run_github.json`)
- **Colab** : `../notebook_colab.ipynb` (1M, 10M interactions, 5 seeds, GPU)

Les fragments divergent (graines de donnees distinctes), puis
`synchroniser.py` les **fusionne par confiance** : chaque etat est evalue sur le
meme held-out, meilleur ECE = plus de poids dans la moyenne ponderee.
Refus automatique si held-out, code_sha ou arch divergent, ou si les fragments
sont des clones (seed_data identique).

## Echange d'etats

Format **bebe-etat/1** (SPEC_ECHANGE.md v1.2, clauses C1-C5) :
`model.pt` + `memoire.npz` + `hist.npz` + `meta.json`.
Ici, dans bebe-ia, l'etat voyage **en commit** (`etats/etat_courant`) : le depot
est la memoire du fragment GitHub. Sur le depot ehwaza/ehwaza, les memes paquets
voyagent **en release assets** — le format est identique, seul le transport
change.

## Reproductibilite

Meme seed = memes resultats (seeds tracees dans `meta.json` : init/data/eval,
config dans chaque `resultats.json`). Chaque run est tracable : seed, source,
nombre d'interactions, debit reel, metriques finales.

## Regles

- Pas de LLM dans la boucle d'apprentissage.
- Code ouvert, lisible, auditable — aucune logique cachee.
- Le held-out ne jamais le regenerer (son SHA est dans meta.json).
- Pull avant push. Un seul push a la fois.
