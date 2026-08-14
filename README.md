# DEEP_LEARNING_EEG — BCIMarcheAvatar

Pipeline de classification EEG pour un système d'interface cerveau-ordinateur (BCI) basé sur l'imagerie motrice de la marche (Gauche / Droite / Marche), développé dans le cadre du projet **PJ102 — BCIMarcheAvatar** (LIO, ÉTS / CRCHUM).

Le projet compare une approche classique de classification EEG (**RLDA**) à deux architectures d'apprentissage profond (**EEGNet**, **DeepConvNet**), appliquées à la fois sur un jeu de données EEG existant (électrodes humides, projet antérieur) et sur un nouveau jeu de données acquis avec le casque **Kaptics** (électrodes sèches).

## Contexte

Les interfaces cerveau-ordinateur sont étudiées comme outil de soutien à la réadaptation motrice. Ce projet évalue si des modèles d'apprentissage profond permettent d'améliorer la reconnaissance des intentions de mouvement par rapport aux méthodes de classification EEG classiques, et comment cette performance varie entre un montage à électrodes humides (données existantes) et un montage à électrodes sèches (casque Kaptics).

## Structure du dépôt
 
```
├── Article BCI/                  # Références bibliographiques et littérature associée
├── classification_RLDA/          # Classification classique : PSD + CSP + LDA régularisé (RLDA)
├── classification_EEGnet/        # Classification par réseau EEGNet (PyTorch)
├── Classification_deepConvnet/   # Classification par réseau DeepConvNet (PyTorch)
└── README.md

## Pipeline commun

1. **Synchronisation** — alignement des enregistrements EEG et des marqueurs d'événements Unity via l'horloge absolue (heure du jour), avec gestion des formats de timestamp AM/PM.
2. **Filtrage** — passe-haut (1 Hz), coupe-bande (60 Hz, bruit secteur), passe-bande (8–30 Hz, bandes mu/beta).
3. **Extraction des epochs** — fenêtre d'imagerie motrice après disparition de la consigne visuelle, avec rejet des artefacts (seuil d'amplitude) et correction de baseline.
4. **Classification** — trois approches indépendantes, validées en **Leave-One-Run-Out (LORO)** :
   - **RLDA** : features PSD par bande, ratios d'asymétrie C3/C4, filtres CSP, sélection des meilleures features (SelectKBest) + LDA régularisé (shrinkage Ledoit-Wolf).
   - **EEGNet** : réseau convolutif compact (convolutions temporelle, spatiale, séparable), entraîné sur le signal normalisé (z-score), avec augmentation de données (bruit gaussien, décalage temporel).
   - **DeepConvNet** : réseau convolutif plus profond (4 blocs, 25→200 filtres), même stratégie d'augmentation de données.

## Résultats

| Méthode | Données existantes (électrodes humides) | Nouvelles données (Kaptics, électrodes sèches) |
|---|---|---|
| RLDA | 78,8 % – 80,0 % | 56,1 % – 65,0 % (moyenne ~60 %) |
| EEGNet | 66,8 % – 73,6 % | 33,6 % – 72,1 % (moyenne 58,6 %) |
| DeepConvNet | 67,0 % – 72,2 % | 52,8 % – 61,4 % (moyenne 56,8 %) |

RLDA surpasse de façon constante les deux approches d'apprentissage profond sur les deux jeux de données, malgré leur plus grande capacité de modélisation — vraisemblablement en raison du nombre limité de données d'entraînement disponibles. Les performances chutent globalement sur les données Kaptics (électrodes sèches) par rapport aux données existantes (électrodes humides), suggérant un signal EEG de moindre qualité avec le système à électrodes sèches.

Le détail complet des résultats (recherche d'hyperparamètres, performances par sujet et par configuration de jours, figures) est disponible dans le mémoire de maîtrise associé.

## Prérequis

```bash
pip install pandas numpy matplotlib scipy scikit-learn torch
```

## Utilisation

Chaque script attend un dossier de sujet contenant des fichiers `runN.csv` (EEG brut) et `runNunity.csv` (marqueurs d'événements) :

```bash
python classification_RLDA.py --subject 01
python eegnet_final.py --subject 01
python deepconvnet.py --subject 01
python analyse_eeg_filtre.py --run run1
```

⚠️ Le chemin racine des données (`BASE_ROOT`) est défini en dur en tête de chaque script et doit être adapté à l'environnement local.

## Données

Les données EEG utilisées dans ce projet (existantes et nouvellement acquises) sont gérées conformément au Système de Gestion de la Qualité du LIO (procédures PQ09, PQ11, PQ12). Elles ne sont pas incluses dans ce dépôt pour des raisons de confidentialité (données de recherche avec participants humains, sous certificat d'éthique).

## Auteure

Emma Carillo — Laboratoire de recherche en imagerie et orthopédie (LIO), ÉTS / CRCHUM
Projet réalisé sous la direction de David Labbé et Cyril Duclos.
