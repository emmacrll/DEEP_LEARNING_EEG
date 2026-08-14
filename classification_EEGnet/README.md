# Classification EEGNet — BCIMarcheAvatar

Ce dossier contient l'application du réseau de neurones **EEGNet** (Lawhern et al., 2018) sur les deux jeux de données EEG du projet **PJ102 — BCIMarcheAvatar** :

- **Données humides** — jeu de données existant, emprunté d'un projet antérieur du LIO (électrodes humides, montage à 19 canaux).
- **Données sèches** — nouvelles données acquises spécifiquement pour ce projet avec le casque **Kaptics** (électrodes sèches, montage à 16 canaux).

## Objectif

Évaluer la performance d'un réseau convolutif compact, spécialement conçu pour l'EEG, pour la classification des intentions de mouvement (Gauche / Droite / Marche) en imagerie motrice, et comparer cette performance selon le type d'électrodes (humides vs sèches) et par rapport aux approches RLDA et DeepConvNet.

## Architecture

EEGNet traite le signal en 3 étapes, pour un nombre de paramètres nettement plus réduit que DeepConvNet :

- **Convolution temporelle** : filtres le long de l'axe temporel (banc de filtres appris)
- **Convolution spatiale (depthwise)** : combine les canaux EEG, similaire dans l'esprit à un CSP appris de bout en bout
- **Convolution séparable** : extrait des motifs temporels plus longs à moindre coût de calcul
- **Classification** : couche dense finale

Chaque bloc applique BatchNorm, activation ELU, average-pooling et dropout (0,25).

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
| Jour 1 | 68,3 % |
| Jour 2 | 66,8 % |
| Jour 3 | 67,2 % |
| Jours 1+2 | 71,8 % |
| Jours 1+3 | 73,6 % |
| Jours 2+3 | 73,0 % |

Les configurations mono-session restent homogènes (66,8–68,3 %). La fusion de plusieurs jours améliore systématiquement les performances, avec un gain jusqu'à 5,3 points pour Jours 1+3 (73,6 %). La variabilité inter-sujet reste modérée (de 65,0 % à 75,8 % selon les sujets en Jours 1+3).

**Données sèches (Kaptics)** — 5 sujets, validation LORO :

| Sujet | Accuracy globale | Écart au hasard (33,3 %) |
|---|---|---|
| Sujet 01 | 60,0 % | +26,7 % |
| Sujet 02 | 60,7 % | +27,4 % |
| Sujet 03 | 72,1 % | +38,8 % |
| Sujet 04 | 66,8 % | +33,5 % |
| Sujet 05 | 33,6 % | +0,2 % |
| **Moyenne (n=5)** | **58,6 %** | **+25,3 %** |

⚠️ Forte variabilité inter-sujet sur les données sèches : un sujet (Sujet 05) est quasiment au niveau du hasard, tandis que les autres dépassent largement le seuil discriminant — cette hétérogénéité est nettement plus marquée que pour RLDA ou DeepConvNet sur le même jeu de données.

## Recherche d'hyperparamètres

Une recherche systématique a été menée par LORO partiel sur trois sujets de référence (Subj04, Subj05, Subj07). La bande bêta seule (13–30 Hz) s'est révélée la plus performante (balanced accuracy moyenne de 55,0 %), suggérant que la désynchronisation bêta pendant l'imagerie motrice et le rebond post-mouvement portent le signal le plus discriminant pour cette architecture — contrairement à DeepConvNet, qui bénéficie davantage d'un signal large bande.

⚠️ **Limite méthodologique** : les hyperparamètres retenus pour les données Kaptics sont ceux identifiés comme optimaux sur les données à électrodes humides — aucune recherche d'hyperparamètres dédiée n'a été menée sur les données sèches, par contrainte de temps. Les performances rapportées sur les données Kaptics doivent être interprétées en tenant compte de cette réserve.

## Comparaison avec les autres approches

Sur les données humides, EEGNet et DeepConvNet obtiennent des performances proches (73,6 % vs 72,2 % en configuration optimale). Sur les données sèches, EEGNet obtient la moyenne la plus élevée des deux approches profondes (58,6 % vs 56,8 %), mais avec une variabilité inter-sujet nettement supérieure, le rendant moins fiable individuellement. Les deux approches restent en retrait par rapport à RLDA (78,8–80,0 % sur données humides ; ~60 % en moyenne sur données sèches), malgré leur plus grande capacité de modélisation — vraisemblablement en raison du nombre limité de données d'entraînement disponibles.

## Structure attendue

```
EEGNet/
├── humide/     # Résultats et scripts pour le jeu de données existant (électrodes humides)
└── seche/      # Résultats et scripts pour le jeu de données Kaptics (électrodes sèches)
```

## Documents qualité associés

- Données humides : PJ102-IDB01, PJ102-DES01, PJ102-LC01, PJ102-LC02
- Données sèches : PJ102-IDB02, PJ102-DES02, PJ102-LC03, PJ102-LC04

## Référence

Lawhern, V. J., et al. (2018). *EEGNet: A compact convolutional neural network for EEG-based brain–computer interfaces.* Journal of Neural Engineering.

## Auteure

Emma Carillo — Laboratoire de recherche en imagerie et orthopédie (LIO), ÉTS / CRCHUM
