# Mode d'emploi — bebe-ia sur Colab

## Ce que fait le notebook (`notebook_colab.ipynb`)

1. **Monte Google Drive** — le corpus Wikipedia et les checkpoints survivent
   aux deconnexions du runtime Colab.
2. **Clone `ehwaza/bebe-ia`** + installe `requirements.txt`.
3. **Config** : `n` (1M, 10M), `seeds` (5 graines), source de donnees, AMP GPU.
4. **Donnees** : Wikipedia (dataset `wikimedia/wikipedia`) telechargee une fois,
   cachee dans `bebe_ia_cache/` sur Drive. Reprise du cache si present.
5. **Boucle** : apprentissage en ligne (1 step SGD par interaction, feedback
   0/1, memoire kNN externe), `torch.cuda.amp` si GPU, checkpoints reguliers
   (`ckpt_every`), **reprise automatique** : relancer la cellule reprend au
   dernier checkpoint (parent_dir).
6. **Evaluation** : ECE sur le held-out fige (`heldout.npz`, jamais regenere),
   accuracy cumulee, courbes sauvegardees.
7. **Synchronisation** : `scripts/synchroniser.py` compare l'etat local au
   fragment GitHub et **fusionne par confiance** (meilleur ECE = plus de poids).

## Lancer un cycle complet

| etape | commande |
|---|---|
| run locale rapide | `python scripts/entrainer_local.py --n 100000 --out resultats/local` |
| 5 seeds | `python scripts/multi_seeds.py --runs 5 --n 50000` |
| fragment GitHub | `python scripts/entrainer_github.py --budget-seconds 18000` (ou le workflow `apprendre.yml`) |
| fusion manuelle | `python scripts/fusionner.py etats/local etats/etat_courant --heldout heldout.npz --out etats/fusionnee` |
| sync complete | `python scripts/synchroniser.py --push` |
| synthese | `python scripts/analyse.py` |

## Reproductibilite

- `meta.json` trace : `seeds.init`, `seeds.data`, `seeds.eval`, `code_sha`,
  `heldout_sha256`, `cycle`, `parent` (format `<fragment>@<code_sha>#<cycle>`).
- Meme seed = memes resultats (torch/numpy/random seedes dans la boucle).
- Chaque run s'ecrit dans son dossier (`seed_<n>/`) avec sa config.

## Points de vigilance

- **heldout.npz est FIGE** : ne jamais le regenerer, son SHA est verifie.
- **Pull avant push**, un seul push a la fois (coordination avec le fragment
  GitHub et le depot ehwaza/ehwaza).
- Colab coupe les runtimes inactifs : les checkpoints sur Drive + la reprise
  automatique sont la pour ca. Une run 10M se fait en plusieurs reprises.
- Le debit est mesure (`ms_per_iter` dans `resultats.json` / `meta.metrics`) :
  pour estimer une run : `n ≈ budget_s × 0.9 / ms_per_iter`.
