# Classification RLDA — BCIMarcheAvatar

Ce dossier contient l'application de la classification **RLDA** (Regularized Linear Discriminant Analysis) sur les deux jeux de données EEG du projet **PJ102 — BCIMarcheAvatar** :

- **Données humides** — jeu de données existant, emprunté d'un projet antérieur du LIO (électrodes humides, montage à 19 canaux).
- **Données sèches** — nouvelles données acquises spécifiquement pour ce projet avec le casque **Kaptics** (électrodes sèches, montage à 16 canaux).

## Objectif

Comparer la performance du classifieur RLDA selon le type d'électrodes (humides vs sèches) et selon les conditions d'acquisition, afin d'évaluer l'impact de la qualité du signal sur la classification des intentions de mouvement (Gauche / Droite / Marche) en imagerie motrice.

## Méthode

Pipeline commun aux deux jeux de données :

1. **Synchronisation** EEG / marqueurs Unity via l'horloge absolue.
2. **Filtrage** : passe-haut (1 Hz), coupe-bande (60 Hz), passe-bande (8–30 Hz, bandes mu/beta).
3. **Extraction des epochs** sur la fenêtre d'imagerie motrice, avec rejet des artefacts et correction de baseline.
4. **Extraction des features** : puissance spectrale (PSD) par canal et par bande de fréquence, ratios d'asymétrie C3/C4, filtres CSP (Common Spatial Patterns), sélection des features les plus discriminantes (SelectKBest).
5. **Classification** : LDA régularisé (shrinkage Ledoit-Wolf, solver eigen).
6. **Validation** : Leave-One-Run-Out (données sèches) / validation croisée à 10 plis stratifiée (données humides).

## Résultats

**Données humides** (9 à 13 sujets selon la configuration, précision moyenne par configuration de jours) :

| Configuration | Précision moyenne |
|---|---|
| Jour 1 | 78,8 % |
| Jour 2 | 79,3 % |
| Jour 3 | 79,6 % |
| Jours 1+2 | 79,1 % |
| Jours 1+3 | 79,1 % |
| Jours 2+3 | 80,0 % |
| Jours 1+2+3 | 79,2 % |

**Données sèches (Kaptics)** — 5 sujets, validation LORO, 10 runs par sujet :

| Sujet | Accuracy globale | Au-dessus du hasard |
|---|---|---|
| Sujet 01 | 60,4 % | +27,0 % |
| Sujet 02 | 61,1 % | +27,7 % |
| Sujet 03 | 56,1 % | +22,7 % |
| Sujet 04 | 65,0 % | +31,7 % |
| Sujet 05 | 57,9 % | +24,5 % |

Les performances sont nettement supérieures au hasard sur les deux jeux de données, mais plus élevées et plus stables sur les données humides (78,8–80,0 %) que sur les données sèches (56,1–65,0 %), suggérant un signal EEG de moindre qualité avec le montage à électrodes sèches.

⚠️ Les hyperparamètres du pipeline (composantes CSP, nombre de features sélectionnées) ont été déterminés par exploration empirique plutôt que par une recherche formelle sur un jeu de validation indépendant — introduisant un léger biais optimiste par rapport à une évaluation strictement indépendante.

## Structure attendue

```
classification RLDA/
├── humide/     # Résultats et scripts pour le jeu de données existant (électrodes humides)
└── seche/      # Résultats et scripts pour le jeu de données Kaptics (électrodes sèches)
```

## Documents qualité associés

- Données humides : PJ102-IDB01, PJ102-DES01, PJ102-LC01, PJ102-LC02
- Données sèches : PJ102-IDB02, PJ102-DES02, PJ102-LC03, PJ102-LC04

## Auteure

Emma Carillo — Laboratoire de recherche en imagerie et orthopédie (LIO), ÉTS / CRCHUM
