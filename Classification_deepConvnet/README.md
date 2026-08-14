# Classification DeepConvNet — BCIMarcheAvatar

Ce dossier contient l'application du réseau de neurones **DeepConvNet** (Schirrmeister et al., 2017) sur les deux jeux de données EEG du projet **PJ102 — BCIMarcheAvatar** :

- **Données humides** — jeu de données existant, emprunté d'un projet antérieur du LIO (électrodes humides, montage à 19 canaux).
- **Données sèches** — nouvelles données acquises spécifiquement pour ce projet avec le casque **Kaptics** (électrodes sèches, montage à 16 canaux).

## Objectif

Évaluer la performance d'un réseau convolutif profond pour la classification des intentions de mouvement (Gauche / Droite / Marche) en imagerie motrice, et comparer cette performance selon le type d'électrodes (humides vs sèches) et par rapport aux approches RLDA et EEGNet.

## Architecture

DeepConvNet empile 4 blocs de convolution qui réduisent progressivement la résolution temporelle (max-pooling) tout en augmentant le nombre de filtres :

- **Bloc 0** : convolution temporelle (25 filtres) + convolution spatiale (25 filtres)
- **Bloc 1** : convolution temporelle (50 filtres)
- **Bloc 2** : convolution temporelle (100 filtres)
- **Bloc 3** : convolution temporelle (200 filtres)
- **Classification** : couche dense finale

Chaque bloc applique BatchNorm, activation ELU, max-pooling et dropout (0,5).

## Méthode

Pipeline commun aux deux jeux de données :

1. **Synchronisation** EEG / marqueurs Unity via l'horloge absolue.
2. **Filtrage** : passe-haut (1 Hz), coupe-bande (60 Hz), passe-bande (8–30 Hz, bandes mu/beta).
3. **Extraction des epochs** avec correction de baseline et normalisation z-score par epoch.
4. **Augmentation de données** (entraînement uniquement) : bruit gaussien (×3) et décalage temporel (×2), soit ×6 le nombre d'epochs d'entraînement.
5. **Entraînement** : Adam, pondération des classes, arrêt anticipé (early stopping) sur la perte de validation.
6. **Validation** : Leave-One-Run-Out (LORO).

## Résultats

**Données humides** (9 sujets, précision moyenne par configuration de jours) :

| Configuration | Précision moyenne |
|---|---|
| Jour 1 | 68,4 % |
| Jour 2 | 67,9 % |
| Jour 3 | 67,0 % |
| Jours 1+2 | 71,1 % |
| Jours 1+3 | 72,2 % |
| Jours 2+3 | 70,5 % |

La fusion de plusieurs jours d'acquisition améliore les performances, la meilleure configuration étant Jours 1+3 (72,2 %).

**Données sèches (Kaptics)** — 5 sujets, validation LORO :

| Sujet | Accuracy globale | Écart au hasard (33,3 %) |
|---|---|---|
| Sujet 01 | 54,2 % | +20,9 % |
| Sujet 02 | 57,1 % | +23,8 % |
| Sujet 03 | 61,4 % | +28,1 % |
| Sujet 04 | 58,6 % | +25,3 % |
| Sujet 05 | 52,8 % | +19,5 % |
| **Moyenne (n=5)** | **56,8 %** | **+23,5 %** |

## Recherche d'hyperparamètres

Une recherche systématique a été menée par LORO partiel sur trois sujets de référence (Subj04, Subj05, Subj07). Contrairement à EEGNet (optimal en bande bêta seule), DeepConvNet performe mieux avec un signal large bande (8–30 Hz) et un taux d'apprentissage de 1×10⁻³, suggérant que cette architecture exploite mieux sa capacité de modélisation avec une information spectrale plus riche.

⚠️ **Limite méthodologique** : les hyperparamètres retenus pour les données Kaptics sont ceux identifiés comme optimaux sur les données à électrodes humides — aucune recherche d'hyperparamètres dédiée n'a été menée sur les données sèches, par contrainte de temps. Les performances rapportées sur les données Kaptics doivent être interprétées en tenant compte de cette réserve.

## Comparaison avec les autres approches

Sur les deux jeux de données, DeepConvNet obtient des performances proches d'EEGNet mais reste en retrait par rapport à RLDA (78,8–80,0 % sur données humides ; ~60 % en moyenne sur données sèches), malgré sa plus grande capacité de modélisation — vraisemblablement en raison du nombre limité de données d'entraînement disponibles.

## Structure attendue

```
DeepConvNet/
├── humide/     # Résultats et scripts pour le jeu de données existant (électrodes humides)
└── seche/      # Résultats et scripts pour le jeu de données Kaptics (électrodes sèches)
```

## Documents qualité associés

- Données humides : PJ102-IDB01, PJ102-DES01, PJ102-LC01, PJ102-LC02
- Données sèches : PJ102-IDB02, PJ102-DES02, PJ102-LC03, PJ102-LC04

## Référence

Schirrmeister, R. T., et al. (2017). *Deep learning with convolutional neural networks for EEG decoding and visualization.* Human Brain Mapping.

## Auteure

Emma Carillo — Laboratoire de recherche en imagerie et orthopédie (LIO), ÉTS / CRCHUM
