#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PIPELINE EEGNET 4 CLASSES — BCI IMAGERIE MOTRICE         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Objectif :                                                                  ║
║  Classer 4 types d'imagerie motrice depuis le signal EEG :                  ║
║    - RIGHT (0) : imaginer le mouvement du pied droit                        ║
║    - LEFT  (1) : imaginer le mouvement du pied gauche                       ║
║    - WALK  (2) : imaginer marcher en avant                                  ║
║    - IDLE  (3) : repos, pas d'imagerie motrice                              ║
║                                                                              ║
║  Architecture hiérarchique en 2 étages :                                    ║
║    1) GATE    : classifieur binaire IDLE vs MOVE                             ║
║    2) EXPERT  : classifieur 3 classes RIGHT / LEFT / WALK                   ║
║                                                                              ║
║  Référence : Lawhern et al. (2018) — EEGNet: A Compact Convolutional        ║
║  Neural Network for EEG-based Brain-Computer Interfaces                     ║
║                                                                              ║
║  Données : fichiers .npz prétraités par bande fréquentielle                 ║
║    - band0812 : bande MU  (8-12Hz)  → entrée du gate                       ║
║    - band1330 : bande BETA (13-30Hz) → combinée avec MU pour l'expert      ║
║                                                                              ║
║  Modes de validation :                                                       ║
║    - LORO (Leave-One-Run-Out) : pour configs mono-jour                      ║
║    - Train/Test split         : pour configs multi-jours                    ║
║                                                                              ║
║  Usage :                                                                     ║
║    # Mode LORO (Jour 1 seul)                                                ║
║    python eegnet_4class.py                                                  ║
║        --npz_mu   SubjXX_Jour1_band0812.npz                                 ║
║        --npz_beta SubjXX_Jour1_band1330.npz                                 ║
║        --idle_windows "nomove1,nomove2"                                     ║
║        --move_windows "move1,move4"                                         ║
║        --move_windows_expert "move1"                                        ║
║        --gate_train_zscore --expert_train_zscore                            ║
║        --epochs 150 --patience 30 --batch_size 32                          ║
║        --expert_aug_factor 5 --seed 42                                      ║
║                                                                              ║
║    # Mode Train/Test (Jour 1 → Jour 2)                                      ║
║    python eegnet_4class.py                                                  ║
║        --npz_mu_train   SubjXX_Jour1_band0812.npz                           ║
║        --npz_beta_train SubjXX_Jour1_band1330.npz                           ║
║        --npz_mu_test    SubjXX_Jour2_band0812.npz                           ║
║        --npz_beta_test  SubjXX_Jour2_band1330.npz                           ║
║        [mêmes autres paramètres]                                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              UTILITAIRES GÉNÉRAUX                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def set_seed(seed: int):
    """
    Fixe tous les seeds aléatoires pour garantir la reproductibilité complète.

    Pourquoi c'est important en EEG :
        Les résultats de classification EEG peuvent varier selon l'initialisation
        aléatoire du réseau. Fixer le seed permet de comparer équitablement
        différentes configurations d'hyperparamètres.

    Args:
        seed : entier pour initialiser numpy, Python random et PyTorch.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def as_run_dict(r):
    """
    Convertit un élément de tableau numpy en dict Python exploitable.

    Pourquoi c'est nécessaire :
        Quand on sauvegarde une liste de dicts dans un .npz avec
        np.savez(runs=np.array(runs, dtype=object)), numpy encapsule
        chaque dict dans un objet numpy. Cette fonction "déplie" cet
        encapsulage pour récupérer le dict Python original.

    Args:
        r : élément du tableau runs (peut être un ndarray object ou un dict).

    Returns:
        dict Python avec les clés y, X_move1, X_nomove1, etc.
    """
    if isinstance(r, np.ndarray):
        return r.item()
    return r


def normalize_runs_array(runs_obj):
    """
    Transforme le tableau numpy de runs en liste Python de dicts exploitables.

    Contexte :
        Les fichiers .npz stockent les runs comme un numpy array d'objets.
        Cette fonction convertit ce format en liste Python standard pour
        faciliter l'itération dans les boucles d'entraînement.

    Args:
        runs_obj : numpy array d'objets contenant les runs EEG.

    Returns:
        list[dict] : liste de runs, chaque run étant un dict Python.
    """
    return [as_run_dict(r) for r in runs_obj]


def parse_windows(s: str):
    """
    Convertit une chaîne de fenêtres temporelles en liste Python.

    Pourquoi des fenêtres multiples :
        Le signal EEG d'un trial couvre ~2 secondes. On peut extraire
        plusieurs sous-fenêtres pour capturer différentes phases de
        l'imagerie motrice. Par exemple "move1,move4" capture le début
        (1050-1300ms) et la fin (1750-2000ms) de l'imagerie.

    Fenêtres disponibles dans les .npz :
        - nomove1 : (251-500)   → repos avant la flèche  → classe IDLE
        - nomove2 : (401-650)   → repos avant la flèche  → classe IDLE
        - move1   : (1051-1300) → début imagerie motrice  → classe MOVE
        - move2   : (1301-1550) → milieu imagerie motrice → classe MOVE
        - move3   : (1501-1750) → suite imagerie motrice  → classe MOVE
        - move4   : (1751-2000) → fin imagerie motrice    → classe MOVE

    Args:
        s : chaîne formatée "move1,move4" ou "nomove1,nomove2".

    Returns:
        list[str] : ["move1", "move4"]
    """
    return [x.strip() for x in s.split(",") if x.strip()]


def fit_channel_zscore(X):
    """
    Calcule les statistiques de normalisation Z-score par canal EEG.

    Pourquoi normaliser par canal :
        Chaque électrode EEG a une amplitude caractéristique différente
        selon sa position sur le scalp et la qualité du contact. Sans
        normalisation, les canaux avec des amplitudes élevées domineraient
        l'apprentissage du réseau. On normalise indépendamment chaque canal
        sur toutes les dimensions (trials + temps) pour ramener tous les
        canaux à mean=0 et std=1.

    IMPORTANT - Anti-leakage :
        Ces statistiques doivent être calculées UNIQUEMENT sur les données
        d'entraînement, puis appliquées aux données de test. Calculer mu/sd
        sur le test introduirait un data leakage.

    Args:
        X : array de shape (n_trials, n_channels, n_time).

    Returns:
        mu : moyenne par canal, shape (1, n_channels, 1).
        sd : écart-type par canal + epsilon, shape (1, n_channels, 1).
    """
    mu = X.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    # epsilon=1e-6 évite la division par zéro sur les canaux plats
    sd = (X.std(axis=(0, 2), keepdims=True) + 1e-6).astype(np.float32)
    return mu, sd


def apply_channel_zscore(X, mu, sd):
    """
    Applique la normalisation Z-score calculée avec fit_channel_zscore.

    Note :
        Toujours appliquer fit sur le train, puis apply sur train ET test.
        Ne jamais recalculer mu/sd sur le test (data leakage).

    Args:
        X  : array à normaliser (n_trials, n_channels, n_time).
        mu : moyenne calculée sur le train.
        sd : écart-type calculé sur le train.

    Returns:
        X normalisé, float32.
    """
    return ((X - mu) / sd).astype(np.float32)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                                  MÉTRIQUES                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def confusion_matrix(y_true, y_pred, n_classes):
    """
    Construit la matrice de confusion entre les vraies et prédites classes.

    Lecture de la matrice :
        - Ligne i = classe vraie i
        - Colonne j = classe prédite j
        - cm[i,j] = nombre de fois où la classe i a été prédite comme j
        - Diagonale = prédictions correctes (idéalement : valeurs élevées)
        - Hors-diagonale = confusions entre classes

    Exemple pour RIGHT/LEFT/WALK/IDLE :
        cm[0,1] = nombre de fois où RIGHT a été confondu avec LEFT

    Args:
        y_true    : labels vrais (array 1D d'entiers).
        y_pred    : labels prédits (array 1D d'entiers).
        n_classes : nombre total de classes (4 pour notre pipeline).

    Returns:
        cm : matrice (n_classes × n_classes) d'entiers.
    """
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def balanced_accuracy(cm):
    """
    Calcule la balanced accuracy (moyenne des recalls par classe).

    Pourquoi balanced_accuracy et pas accuracy simple en EEG/BCI :
        Notre dataset est déséquilibré — IDLE représente ~50% des trials
        (car on ajoute 2 fenêtres nomove par trial actif). Un classifieur
        naïf qui prédit toujours IDLE atteindrait 50% d'accuracy mais
        serait complètement inutile. La balanced accuracy donne un poids
        égal à chaque classe et révèle ce biais.

        Exemple : si IDLE est toujours correct mais RIGHT/LEFT/WALK sont
        aléatoires → accuracy = 0.65 mais balanced_accuracy = 0.375

    Formule :
        balanced_accuracy = (1/n_classes) × Σ (cm[i,i] / Σ cm[i,:])

    Args:
        cm : matrice de confusion (n_classes × n_classes).

    Returns:
        float entre 0 et 1. Hasard = 1/n_classes = 0.25 (4 classes).
    """
    rec = []
    for i in range(cm.shape[0]):
        denom = cm[i].sum()  # total trials de la classe i
        # recall de la classe i = vrais positifs / total classe i
        rec.append(cm[i, i] / denom if denom > 0 else 0.0)
    return float(np.mean(rec))


def print_cm(cm, labels):
    """
    Affiche la matrice de confusion avec les noms des classes.

    Format :
        Colonnes = classes prédites
        Lignes   = classes vraies

    Args:
        cm     : matrice de confusion numpy.
        labels : liste des noms de classes ["RIGHT","LEFT","WALK","IDLE"].
    """
    print("        " + "  ".join([f"{l:>6s}" for l in labels]))
    for i, l in enumerate(labels):
        row = "  ".join([f"{v:6d}" for v in cm[i]])
        print(f"{l:>6s}  {row}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           SPLIT STRATIFIÉ                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def stratified_split_indices(y, val_frac, seed):
    """
    Divise les indices en train/validation en respectant les proportions de classes.

    Pourquoi stratifié :
        Si on tire aléatoirement sans stratification, certains folds pourraient
        avoir très peu de trials d'une classe rare. En EEG avec ~80 trials par
        classe, c'est fréquent. La stratification garantit que chaque split
        contient la même proportion de chaque classe.

    Exemple :
        y = [0,0,0,1,1,1,2,2,2], val_frac=0.33
        → train = [0,0,1,1,2,2], val = [0,1,2]
        → chaque classe représentée équitablement dans val

    Args:
        y        : array de labels (entiers).
        val_frac : fraction des données pour la validation (ex: 0.2 = 20%).
        seed     : graine aléatoire pour reproductibilité.

    Returns:
        tr_idx  : indices d'entraînement.
        val_idx : indices de validation.
    """
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    val_idx, tr_idx = [], []

    for c in np.unique(y):
        # indices de tous les trials de la classe c
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        # calcul du nombre de trials pour la validation
        n_val = int(round(len(idx) * val_frac))
        # garantir au moins 1 trial en val si possible, et au moins 1 en train
        if len(idx) >= 2:
            n_val = max(1, min(n_val, len(idx) - 1))
        else:
            n_val = 0
        val_idx.append(idx[:n_val])
        tr_idx.append(idx[n_val:])

    val_idx = np.concatenate(val_idx) if val_idx else np.array([], dtype=int)
    tr_idx  = np.concatenate(tr_idx)  if tr_idx  else np.array([], dtype=int)
    # mélange final pour éviter que les classes soient groupées
    rng.shuffle(val_idx)
    rng.shuffle(tr_idx)
    return tr_idx, val_idx


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         AUGMENTATION DE DONNÉES                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def augment_gaussian_noise(X, y, w, rng, noise_factor=0.05):
    """
    Augmente les données EEG en ajoutant du bruit gaussien réaliste.

    Pourquoi ce type d'augmentation en EEG :
        Le signal EEG naturel est bruité. En ajoutant du bruit gaussien
        proportionnel à l'écart-type du signal, on simule la variabilité
        naturelle du signal entre les essais. Cela empêche le réseau de
        mémoriser les trials exacts et améliore la généralisation.

        Le bruit est proportionnel à la std du trial (bruit = 5% de la std)
        pour que l'amplitude du bruit reste réaliste par rapport au signal.

    Args:
        X            : données EEG (n_trials, n_channels, n_time).
        y            : labels (n_trials,).
        w            : poids par trial (n_trials,).
        rng          : générateur aléatoire numpy.
        noise_factor : fraction de la std ajoutée comme bruit (défaut 5%).

    Returns:
        X augmenté, y copié, w copié.
    """
    # std calculée par trial (sur tous les canaux et tous les points temporels)
    std   = X.std(axis=(1, 2), keepdims=True)
    # bruit gaussien de même forme que X, scaled par la std du trial
    noise = rng.randn(*X.shape).astype(np.float32) * std * noise_factor
    return (X + noise).astype(np.float32), y.copy(), w.copy()


def augment_time_shift(X, y, w, rng, max_shift=15):
    """
    Augmente les données EEG en décalant légèrement les signaux dans le temps.

    Pourquoi ce type d'augmentation en EEG :
        Le moment exact du début de l'imagerie motrice varie légèrement d'un
        essai à l'autre (latence de réaction). En simulant des décalages
        temporels de ±15 samples (±58ms à 256Hz), on apprend au réseau à
        être robuste à ces variations temporelles naturelles.

        max_shift=15 samples = 58ms → plausible physiologiquement.

    Args:
        X         : données EEG (n_trials, n_channels, n_time).
        y         : labels.
        w         : poids.
        rng       : générateur aléatoire numpy.
        max_shift : décalage maximum en samples (défaut 15 = 58ms à 256Hz).

    Returns:
        X_aug décalé, y copié, w copié.
    """
    X_aug = np.zeros_like(X)
    for i in range(len(X)):
        shift = rng.randint(-max_shift, max_shift + 1)
        if shift > 0:
            # décalage vers la droite : padding de zéros à gauche
            X_aug[i, :, shift:] = X[i, :, :-shift]
        elif shift < 0:
            # décalage vers la gauche : padding de zéros à droite
            X_aug[i, :, :shift] = X[i, :, -shift:]
        else:
            # pas de décalage
            X_aug[i] = X[i]
    return X_aug.astype(np.float32), y.copy(), w.copy()


def augment_expert_data(X, y, aug_factor, seed=42):
    """
    Augmente les données d'entraînement de l'expert avec pondération adaptative.

    Stratégie d'augmentation :
        On alterne entre bruit gaussien (pairs) et décalage temporel (impairs)
        pour créer aug_factor versions augmentées des données originales.
        Le dataset final contient (aug_factor + 1) × n_trials_originaux.

    Pondération par qualité de signal :
        Les trials EEG de bonne qualité (variance > seuil) reçoivent un
        poids plus élevé (1.0) que les trials bruités ou plats (0.2).
        Cela guide l'apprentissage vers les patterns discriminants fiables.

        Note : un trial avec variance < 1e-4 µV² est probablement plat
        (mauvais contact électrode) et ne devrait pas influencer le modèle.

    Args:
        X          : données EEG (n_trials, n_channels, n_time).
        y          : labels.
        aug_factor : nombre de versions augmentées à créer (défaut 5 → ×6).
        seed       : graine aléatoire.

    Returns:
        X_aug : données augmentées (n_trials × (aug_factor+1), n_channels, n_time).
        y_aug : labels augmentés.
        w_aug : poids par trial (pondération signal + augmentation).
    """
    # calcul de la variance par trial sur tous les canaux et points temporels
    trial_var = X.var(axis=(1, 2))
    # poids : 1.0 pour signal normal, 0.2 pour signal plat/bruité
    w = np.where(trial_var > 1e-4, 1.0, 0.2).astype(np.float32)
    n_low = (w < 1.0).sum()
    if n_low > 0:
        print(f"    [EXPERT_WEIGHTS] {n_low}/{len(w)} trials poids faible (0.2)")

    rng = np.random.RandomState(seed)
    X_list, y_list, w_list = [X], [y], [w]

    for i in range(aug_factor):
        # alternance bruit / décalage pour maximiser la diversité
        if i % 2 == 0:
            Xa, ya, wa = augment_gaussian_noise(X, y, w, rng)
        else:
            Xa, ya, wa = augment_time_shift(X, y, w, rng)
        X_list.append(Xa)
        y_list.append(ya)
        w_list.append(wa)

    return (
        np.concatenate(X_list).astype(np.float32),
        np.concatenate(y_list).astype(np.int64),
        np.concatenate(w_list).astype(np.float32)
    )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         CONSTRUCTION DES FENÊTRES                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def stack_windows(run, windows, mode="time"):
    """
    Extrait et empile plusieurs fenêtres temporelles d'un run EEG.

    Deux modes d'empilement :
        - "time"   : concatène les fenêtres sur l'axe temporel
                     move1 (250 samples) + move4 (250 samples) = 500 samples
                     → le réseau voit le début ET la fin de l'imagerie

        - "trials" : concatène les fenêtres sur l'axe des trials
                     → multiplie le nombre de trials mais garde 250 samples chacun

    Pourquoi le mode "time" pour le gate :
        Concaténer move1+move4 donne plus de contexte temporel au gate.
        Il peut observer l'évolution du signal pendant toute la période
        d'imagerie pour mieux distinguer IDLE de MOVE.

    Args:
        run     : dict du run contenant X_move1, X_move4, etc.
        windows : liste de noms de fenêtres ["move1", "move4"].
        mode    : "time" ou "trials".

    Returns:
        X : array empilé (n_trials, n_channels, n_time_concat).
        y : labels du run (entiers).

    Raises:
        KeyError       : si une fenêtre demandée n'existe pas dans le run.
        RuntimeError   : si les dimensions ne correspondent pas.
        ValueError     : si mode invalide.
    """
    run = as_run_dict(run)
    y   = np.array(run["y"]).astype(int)
    X_list = []

    for w in windows:
        key = f"X_{w}"
        if key not in run:
            raise KeyError(f"Fenêtre '{key}' introuvable dans le run. "
                          f"Fenêtres disponibles : {[k for k in run.keys() if k.startswith('X_')]}")
        Xw = np.array(run[key]).astype(np.float32)
        # vérification cohérence nombre de trials
        if Xw.shape[0] != len(y):
            raise RuntimeError(f"{key} : {Xw.shape[0]} trials ≠ {len(y)} labels")
        X_list.append(Xw)

    if mode == "time":
        # concaténation sur l'axe 2 (temps) : (n, ch, t1+t2+...)
        return np.concatenate(X_list, axis=2).astype(np.float32), y.astype(int)
    if mode == "trials":
        # concaténation sur l'axe 0 (trials) : (n*k, ch, t)
        return (
            np.concatenate(X_list, axis=0).astype(np.float32),
            np.concatenate([y] * len(X_list)).astype(int)
        )
    raise ValueError(f"mode doit être 'time' ou 'trials', reçu : '{mode}'")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                       CONSTRUCTION DES DATASETS                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def build_gate_idle_vs_move(run_mu, idle_windows, move_windows, stack_mode):
    """
    Construit le dataset binaire pour entraîner le modèle GATE.

    Le gate doit apprendre à distinguer :
        - IDLE (label 0) : sujet au repos, aucune imagerie motrice
        - MOVE (label 1) : sujet en train d'imaginer un mouvement

    Source des données :
        - IDLE → fenêtres nomove1 et nomove2 (signal avant la flèche)
        - MOVE → fenêtres move1 et/ou move4 (signal pendant l'imagerie)

    Note sur les labels :
        Les labels originaux RIGHT=1, LEFT=2, WALK=3 sont tous remplacés
        par MOVE=1 pour le gate. Le gate n'a pas besoin de distinguer
        quel type de mouvement, juste s'il y a un mouvement ou non.

    Args:
        run_mu       : run EEG en bande MU (8-12Hz).
        idle_windows : fenêtres repos ["nomove1", "nomove2"].
        move_windows : fenêtres mouvement ["move1", "move4"].
        stack_mode   : "time" pour concaténation temporelle.

    Returns:
        X : (n_idle + n_move, n_channels, n_time), float32.
        y : labels binaires 0=IDLE, 1=MOVE, int64.
    """
    # extraction fenêtres repos → tous labelisés IDLE=0
    X_idle, _ = stack_windows(run_mu, idle_windows, mode=stack_mode)
    # extraction fenêtres mouvement → tous labelisés MOVE=1
    X_move, _ = stack_windows(run_mu, move_windows, mode=stack_mode)

    y_idle = np.zeros(X_idle.shape[0], dtype=np.int64)   # 0 = IDLE
    y_move = np.ones(X_move.shape[0],  dtype=np.int64)   # 1 = MOVE

    return (
        np.concatenate([X_idle, X_move]).astype(np.float32),
        np.concatenate([y_idle, y_move]).astype(np.int64)
    )


def build_expert_rl(run_mu, run_beta, move_windows, stack_mode):
    """
    Construit le dataset 3 classes pour entraîner le modèle EXPERT.

    L'expert doit apprendre à distinguer les 3 types de mouvements :
        - RIGHT (0) : imagerie pied droit
        - LEFT  (1) : imagerie pied gauche
        - WALK  (2) : imagerie marche

    Fusion MU + BETA :
        L'expert combine les deux bandes fréquentielles en concaténant
        les canaux MU et BETA sur l'axe des canaux :
            shape MU   : (n, 8, n_time)
            shape BETA : (n, 8, n_time)
            shape final : (n, 16, n_time)  ← 16 canaux = 8 MU + 8 BETA

        Pourquoi MU + BETA :
            Les patterns ERD/ERS de l'imagerie motrice apparaissent dans les
            deux bandes. MU (8-12Hz) code l'inhibition motrice, BETA (13-30Hz)
            code le rebond post-mouvement. Combiner les deux enrichit la
            représentation apprise par le réseau.

    Remappage des labels :
        Les labels originaux (1=RIGHT, 2=LEFT, 3=WALK) sont remappés en
        (0=RIGHT, 1=LEFT, 2=WALK) pour la convention PyTorch (classes 0-indexées).

    Note :
        Les trials IDLE sont exclus — l'expert ne voit que les mouvements
        car le gate a déjà filtré les IDLE en amont du pipeline.

    Args:
        run_mu    : run EEG bande MU  (8-12Hz).
        run_beta  : run EEG bande BETA (13-30Hz).
        move_windows : fenêtres mouvement ["move1"].
        stack_mode   : "time".

    Returns:
        X : (n_move, 16, n_time), float32. 16 = 8 canaux MU + 8 canaux BETA.
        y : labels 3 classes (0=RIGHT, 1=LEFT, 2=WALK), int64.
    """
    # extraction fenêtres mouvement pour MU
    Xm, y  = stack_windows(run_mu,   move_windows, mode=stack_mode)
    # extraction fenêtres mouvement pour BETA
    Xb, yb = stack_windows(run_beta, move_windows, mode=stack_mode)

    # vérification cohérence labels entre les deux bandes
    if not np.array_equal(y, yb):
        raise RuntimeError("Labels MU et BETA ne correspondent pas — problème dans les fichiers .npz")

    # concaténation MU + BETA sur l'axe des canaux : (n, 8, t) → (n, 16, t)
    X = np.concatenate([Xm, Xb], axis=1).astype(np.float32)

    # filtrage : garder uniquement les classes actives (1=RIGHT, 2=LEFT, 3=WALK)
    # les éventuels labels 0 (IDLE) dans le fichier sont exclus
    mask = (y == 1) | (y == 2) | (y == 3)
    X = X[mask]
    y = y[mask]

    # remappage 0-indexé pour PyTorch CrossEntropyLoss
    # 1 → 0 (RIGHT), 2 → 1 (LEFT), 3 → 2 (WALK)
    y_new = np.zeros_like(y)
    y_new[y == 2] = 1  # LEFT
    y_new[y == 3] = 2  # WALK
    # y == 1 reste 0 (RIGHT)

    return X, y_new.astype(np.int64)


def build_test_eval_4class(run_mu, run_beta,
                           idle_windows, move_windows_gate,
                           move_windows_expert,
                           stack_mode_gate, stack_mode_expert):
    """
    Prépare les données d'évaluation pour le pipeline complet 4 classes.

    Cette fonction construit DEUX entrées séparées :
        1. Xg : entrée du GATE (bande MU, fenêtres move1+move4 + nomove)
        2. Xe : entrée de l'EXPERT (bandes MU+BETA, fenêtres move1 seulement)

    Structure du pipeline d'évaluation :
        IDLE trials  → gate prédit IDLE    → y_pred = 3 (IDLE)
        MOVE trials  → gate prédit MOVE    → expert prédit RIGHT/LEFT/WALK

    Gestion des labels 4 classes :
        Dans les fichiers .npz, les labels sont :
            1=RIGHT, 2=LEFT, 3=WALK
        Pour l'évaluation 4 classes, on remappage en :
            0=RIGHT, 1=LEFT, 2=WALK, 3=IDLE
        (IDLE détecté par le gate est assigné à la classe 3)

    Args:
        run_mu, run_beta    : runs EEG en bande MU et BETA.
        idle_windows        : fenêtres repos pour le gate.
        move_windows_gate   : fenêtres mouvement pour le gate ["move1","move4"].
        move_windows_expert : fenêtres mouvement pour l'expert ["move1"].
        stack_mode_gate/expert : mode d'empilement "time".

    Returns:
        Xg        : entrée gate (n_total, n_ch_gate, n_time_gate).
        Xe        : entrée expert (n_total, 16, n_time_expert).
        y_true    : vrais labels 4 classes (0=RIGHT,1=LEFT,2=WALK,3=IDLE).
        idle_mask : masque booléen, True pour les trials IDLE.
        move_mask : masque booléen, True pour les trials MOVE.
    """
    # ── PARTIE IDLE ──
    # extraction des trials repos → ils seront tous IDLE=3 dans l'évaluation
    Xg_idle, _ = stack_windows(run_mu, idle_windows, mode=stack_mode_gate)
    n_idle      = Xg_idle.shape[0]
    y_idle_true = np.full(n_idle, 3, dtype=np.int64)  # 3 = IDLE

    # ── PARTIE MOVE — GATE ──
    # extraction des trials actifs pour le gate
    Xg_move, y_move_raw = stack_windows(run_mu, move_windows_gate, mode=stack_mode_gate)
    # filtrage des labels valides (au cas où il y aurait des labels 0 résiduels)
    mask_123   = (y_move_raw == 1) | (y_move_raw == 2) | (y_move_raw == 3)
    Xg_move    = Xg_move[mask_123]
    y_move_raw = y_move_raw[mask_123]
    # remappage 4 classes : 1→0 (RIGHT), 2→1 (LEFT), 3→2 (WALK)
    y_move_true = np.full(len(y_move_raw), -1, dtype=np.int64)
    y_move_true[y_move_raw == 1] = 0  # RIGHT
    y_move_true[y_move_raw == 2] = 1  # LEFT
    y_move_true[y_move_raw == 3] = 2  # WALK

    # ── PARTIE MOVE — EXPERT ──
    # extraction fenêtres pour l'expert (MU + BETA concaténés sur canaux)
    Xme, ye  = stack_windows(run_mu,   move_windows_expert, mode=stack_mode_expert)
    Xbe, ybe = stack_windows(run_beta, move_windows_expert, mode=stack_mode_expert)
    if not np.array_equal(ye, ybe):
        raise RuntimeError("Labels MU/BETA expert ne correspondent pas")
    # fusion MU + BETA → (n, 16, n_time)
    X_expert_all = np.concatenate([Xme, Xbe], axis=1).astype(np.float32)
    # filtrage trials actifs uniquement
    _, y_exp_raw = stack_windows(run_mu, move_windows_expert, mode=stack_mode_expert)
    mask_exp      = (y_exp_raw == 1) | (y_exp_raw == 2) | (y_exp_raw == 3)
    X_expert_move = X_expert_all[mask_exp]

    # vérification alignement entre gate et expert (même nombre de trials MOVE)
    if X_expert_move.shape[0] != Xg_move.shape[0]:
        raise RuntimeError(
            f"Désalignement gate ({Xg_move.shape[0]} trials) "
            f"vs expert ({X_expert_move.shape[0]} trials)"
        )

    # ── ASSEMBLAGE FINAL ──
    # input gate = [trials IDLE | trials MOVE]
    Xg = np.concatenate([Xg_idle, Xg_move]).astype(np.float32)

    # input expert = [zéros pour IDLE (dummy) | features MOVE réelles]
    # Note : les zéros pour IDLE ne seront jamais utilisés par l'expert
    # car le gate aura déjà identifié ces trials comme IDLE
    Xe_dum = np.zeros((n_idle, X_expert_move.shape[1], X_expert_move.shape[2]),
                      dtype=np.float32)
    Xe = np.concatenate([Xe_dum, X_expert_move]).astype(np.float32)

    # labels finaux 4 classes
    y_true = np.concatenate([y_idle_true, y_move_true]).astype(np.int64)

    # masques pour savoir quels trials sont IDLE/MOVE lors de la prédiction
    idle_mask = np.zeros(len(y_true), dtype=bool)
    idle_mask[:n_idle] = True

    return Xg, Xe, y_true, idle_mask, ~idle_mask


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                          ARCHITECTURE EEGNET                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class EEGNet(nn.Module):
    """
    Architecture EEGNet pour la classification de signaux EEG.

    Référence : Lawhern et al. (2018) — EEGNet: A Compact Convolutional
    Neural Network for EEG-based Brain-Computer Interfaces.

    Architecture en 3 blocs :

    1) BLOC TEMPOREL (self.temporal) :
       Conv2d(1 → F1, kernel=(1, kernel_len))
       + BatchNorm
       → Apprend des filtres temporels indépendants pour chaque canal.
         kernel_len=64 captures des patterns dans ~250ms (à 256Hz).
         kernel_len=128 → ~500ms, kernel_len=192 → ~750ms.
         Un kernel plus long capture des oscillations plus lentes (alpha, beta).

    2) BLOC SPATIAL (self.spatial) :
       Conv2d(F1 → F1*D, kernel=(n_ch, 1), groups=F1)
       + BatchNorm + ELU + AvgPool + Dropout
       → Apprend des combinaisons linéaires entre canaux EEG.
         groups=F1 = convolution "depthwise" : chaque filtre temporel
         a sa propre convolution spatiale. Cela réduit les paramètres.
         AvgPool(1,4) : sous-échantillonnage ×4 dans le temps.

    3) BLOC SÉPARABLE (self.separable) :
       Conv2d depthwise + Conv2d pointwise (1×1)
       + BatchNorm + ELU + AvgPool + Dropout
       → Raffinement des features avec peu de paramètres.
         AvgPool(1,8) : sous-échantillonnage ×8 supplémentaire.

    4) CLASSIFIEUR DYNAMIQUE :
       Linear(n_features → n_classes)
       → Créé au premier forward car la dimension dépend de n_time.
         n_time=250 → features ≈ F2 × (250/4/8) ≈ 16 × 7 = 112
         n_time=500 → features ≈ 16 × 15 = 240

    Paramètres typiques :
        F1=8, D=2, F2=16, kernel_len=64, dropout=0.25
        Total paramètres ≈ 2000-5000 selon n_ch et n_time
        (très compact pour EEG avec peu de données)

    Args:
        n_ch       : nombre de canaux EEG (8 pour notre pipeline, 16 pour expert MU+BETA).
        n_classes  : nombre de classes (2 pour gate, 3 pour expert).
        F1         : nombre de filtres temporels (défaut 8).
        D          : profondeur spatiale — F1*D filtres après bloc spatial (défaut 2).
        F2         : nombre de filtres séparables (défaut 16).
        kernel_len : longueur du filtre temporel en samples (défaut 64 = 250ms à 256Hz).
        dropout    : taux de dropout pour régularisation (défaut 0.25).
    """
    def __init__(self, n_ch, n_classes=2, F1=8, D=2, F2=16,
                 kernel_len=64, dropout=0.25):
        super().__init__()
        self.n_classes = n_classes

        # Bloc 1 : Conv temporelle
        # Input  : (batch, 1, n_ch, n_time)
        # Output : (batch, F1, n_ch, n_time)  [même taille grâce au padding]
        self.temporal = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_len),
                      padding=(0, kernel_len // 2), bias=False),
            nn.BatchNorm2d(F1)
        )

        # Bloc 2 : Conv spatiale (depthwise)
        # Input  : (batch, F1, n_ch, n_time)
        # Output : (batch, F1*D, 1, n_time/4)  [n_ch → 1 grâce à kernel (n_ch,1)]
        self.spatial = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),   # sous-échantillonnage ×4 dans le temps
            nn.Dropout(dropout)
        )

        # Bloc 3 : Conv séparable (depthwise + pointwise)
        # Input  : (batch, F1*D, 1, n_time/4)
        # Output : (batch, F2, 1, n_time/32)
        self.separable = nn.Sequential(
            # depthwise : un filtre par canal
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8),
                      groups=F1 * D, bias=False),
            # pointwise : combinaison entre canaux
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),   # sous-échantillonnage ×8 supplémentaire
            nn.Dropout(dropout)
        )

        # Classifieur final (créé dynamiquement au premier forward)
        # car sa taille dépend de n_time et n_ch
        self.classifier = None

    def forward(self, x):
        """
        Passe avant du réseau.

        Flow des dimensions (exemple : batch=32, n_ch=8, n_time=250, F1=8, D=2, F2=16) :
            Input  : (32, 8, 250)
            Unsqueeze : (32, 1, 8, 250)       ← ajout dim canal CNN
            Temporal  : (32, 8, 8, 250)       ← F1=8 filtres temporels
            Spatial   : (32, 16, 1, 62)       ← F1*D=16, AvgPool/4
            Separable : (32, 16, 1, ~7)       ← F2=16, AvgPool/8
            Flatten   : (32, ~112)            ← 16 × 7 features
            Linear    : (32, n_classes)       ← classification finale
        """
        # ajout de la dimension "canal CNN" requise par Conv2d
        x = x.unsqueeze(1)

        # extraction des features EEGNet
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.separable(x)

        # aplatissement pour le classifieur linéaire
        x = x.flatten(1)

        # création du classifieur au premier appel (taille inconnue à l'avance)
        if self.classifier is None:
            self.classifier = nn.Linear(x.shape[1], self.n_classes).to(x.device)

        return self.classifier(x)


class SimpleDataset(Dataset):
    """
    Dataset PyTorch simple sans pondération.
    Utilisé pour le gate (pas de pondération par qualité de signal).
    """
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class WeightedDataset(Dataset):
    """
    Dataset PyTorch avec poids par trial.

    Utilisé pour l'expert car certains trials sont pondérés différemment
    selon la qualité du signal et l'augmentation de données.

    Les poids sont utilisés dans la loss pondérée :
        loss = mean(CrossEntropy(pred, true) × w)
    → Les trials avec w=1.0 contribuent plus à l'apprentissage que w=0.2.
    """
    def __init__(self, X, y, w):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.w = torch.from_numpy(w).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i], self.w[i]


@torch.no_grad()
def predict_logits(model, X, device, batch_size=256):
    """
    Effectue une prédiction par batch sans calcul de gradient.

    Pourquoi par batch :
        Pour les grands ensembles de données, passer tout X d'un coup
        peut saturer la mémoire GPU/CPU. Le traitement par batch de 256
        est un bon compromis vitesse/mémoire.

    @torch.no_grad() :
        Désactive le calcul du graphe de gradient — indispensable en
        inférence pour économiser mémoire et accélérer le calcul.

    Args:
        model      : modèle EEGNet en mode eval.
        X          : données à prédire (n_trials, n_ch, n_time).
        device     : 'cpu' ou 'cuda'.
        batch_size : taille des batchs d'inférence.

    Returns:
        logits : tensor (n_trials, n_classes) — scores bruts avant softmax.
    """
    model.eval()
    dl = DataLoader(
        TensorDataset(torch.from_numpy(X).float()),
        batch_size=batch_size,
        shuffle=False  # important : ne pas mélanger pour garder l'ordre
    )
    outs = []
    for (xb,) in dl:
        outs.append(model(xb.to(device)).detach().cpu())
    return torch.cat(outs, dim=0)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                        ENTRAÎNEMENT DU GATE                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def train_gate(X_all, y_all, device, *, n_classes, val_split, seed_split,
               do_zscore, kernel_len, dropout, lr, wd, batch_size,
               epochs, patience, F1, D, F2):
    """
    Entraîne le modèle GATE (classifieur binaire IDLE vs MOVE).

    Le gate est le premier étage du pipeline hiérarchique. Il doit
    apprendre à détecter si le sujet est en train d'imaginer un mouvement
    ou non, sans se préoccuper du type de mouvement.

    Pipeline d'entraînement :
        1. Split stratifié train/val (20% val par défaut)
        2. Z-score normalisé sur le train uniquement
        3. Initialisation EEGNet (n_classes=2)
        4. Entraînement avec CrossEntropyLoss
        5. Early stopping basé sur balanced_accuracy (pas accuracy brute)
           → évite de s'arrêter trop tôt si le gate est biaisé vers IDLE
        6. Restauration du meilleur état sauvegardé

    Kernel temporel :
        kernel_len=128 (500ms) pour le gate — plus long que pour l'expert
        car distinguer IDLE/MOVE nécessite de voir l'évolution globale
        du signal sur une longue période (présence ou absence d'activité).

    Args:
        X_all, y_all : données et labels (IDLE=0, MOVE=1).
        device       : 'cpu' ou 'cuda'.
        [hyperparamètres] : voir argparse dans main().

    Returns:
        model : EEGNet entraîné.
        mu, sd : statistiques de normalisation (None si do_zscore=False).
    """
    # ── 1. SPLIT TRAIN / VALIDATION ──
    tr_idx, va_idx = stratified_split_indices(y_all, val_split, seed_split)
    # fallback si val trop petit (< 1 trial par classe)
    if len(va_idx) == 0:
        rng = np.random.RandomState(seed_split)
        idx = np.arange(len(y_all))
        rng.shuffle(idx)
        n_val  = max(1, int(0.05 * len(idx)))
        va_idx = idx[:n_val]
        tr_idx = idx[n_val:]

    Xtr, ytr = X_all[tr_idx], y_all[tr_idx]
    Xva, yva = X_all[va_idx], y_all[va_idx]

    # ── 2. NORMALISATION Z-SCORE ──
    # CRITIQUE : calculer mu/sd UNIQUEMENT sur le train
    mu = sd = None
    if do_zscore:
        mu, sd = fit_channel_zscore(Xtr)
        Xtr = apply_channel_zscore(Xtr, mu, sd)
        Xva = apply_channel_zscore(Xva, mu, sd)

    # ── 3. INITIALISATION DU MODÈLE ──
    model   = EEGNet(n_ch=Xtr.shape[1], n_classes=n_classes, F1=F1, D=D, F2=F2,
                     kernel_len=kernel_len, dropout=dropout).to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()
    dl_tr   = DataLoader(SimpleDataset(Xtr, ytr),
                         batch_size=batch_size, shuffle=True)

    # ── 4. BOUCLE D'ENTRAÎNEMENT AVEC EARLY STOPPING ──
    best_bal      = -1.0    # meilleure balanced accuracy sur la validation
    best_state    = None    # poids du meilleur modèle
    patience_left = patience

    for _ in range(epochs):
        # ── Phase train ──
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)   # plus efficace que zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

        # ── Phase validation ──
        with torch.no_grad():
            pred = torch.argmax(
                predict_logits(model, Xva, device), dim=1
            ).numpy()
        bal = balanced_accuracy(confusion_matrix(yva, pred, n_classes))

        # ── Early stopping ──
        if bal > best_bal + 1e-6:
            # amélioration → sauvegarder l'état du modèle
            best_bal   = bal
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break   # arrêt anticipé

    # ── 5. RESTAURATION DU MEILLEUR MODÈLE ──
    if best_state:
        model.load_state_dict(best_state)

    return model, mu, sd


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                        ENTRAÎNEMENT DE L'EXPERT                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def train_expert(X_all, y_all, device, *, n_classes, val_split, seed_split,
                 do_zscore, kernel_len, dropout, lr, wd, batch_size,
                 epochs, patience, F1, D, F2, aug_factor=9, we_all=None):
    """
    Entraîne le modèle EXPERT (classifieur 3 classes RIGHT / LEFT / WALK).

    L'expert est le second étage du pipeline. Il reçoit uniquement les
    segments détectés comme MOVE par le gate et doit distinguer quel
    type de mouvement est imaginé.

    Différences par rapport au gate :
        1. Données : MU+BETA concaténés → 16 canaux au lieu de 8
        2. Augmentation : aug_factor=9 → ×10 les trials (peu de données)
        3. Loss pondérée : certains trials ont moins de poids si bruités
        4. kernel_len=192 → 750ms : captures des patterns plus fins

    Pourquoi l'expert est plus difficile que le gate :
        - Gate : IDLE vs MOVE → différence spectrale forte (alpha/beta power)
        - Expert : RIGHT vs LEFT vs WALK → différences subtiles d'asymétrie
          C3/C4 et de topographie. Beaucoup plus difficile avec peu de données.

    Args:
        X_all, y_all : données et labels (0=RIGHT, 1=LEFT, 2=WALK).
        we_all       : poids par trial (optionnel), priorité aux trials récents.
        aug_factor   : nombre de versions augmentées (défaut 9 → ×10).
        [autres hyperparamètres] : identiques à train_gate.

    Returns:
        model : EEGNet entraîné.
        mu, sd : statistiques de normalisation.
    """
    # ── 1. SPLIT STRATIFIÉ ──
    tr_idx, va_idx = stratified_split_indices(y_all, val_split, seed_split)
    if len(va_idx) == 0:
        rng   = np.random.RandomState(seed_split)
        idx   = np.arange(len(y_all))
        rng.shuffle(idx)
        n_val  = max(1, int(0.05 * len(idx)))
        va_idx = idx[:n_val]
        tr_idx = idx[n_val:]

    Xtr, ytr = X_all[tr_idx], y_all[tr_idx]
    Xva, yva = X_all[va_idx], y_all[va_idx]

    # ── 2. POIDS DES RUNS ──
    # poids par run (runs récents peuvent avoir weight=1.5 si recalibration)
    we_tr = (we_all[tr_idx] if we_all is not None
             else np.ones(len(ytr), dtype=np.float32))

    # ── 3. AUGMENTATION DE DONNÉES ──
    # CRUCIAL pour l'expert : ~165 trials/classe → trop peu sans augmentation
    if aug_factor > 0:
        n_before = len(ytr)
        Xtr, ytr, w_aug = augment_expert_data(
            Xtr, ytr, aug_factor=aug_factor, seed=seed_split + 999
        )
        print(f"    [EXPERT_AUG] {n_before} → {len(ytr)} trials (x{aug_factor+1})")
        # dupliquer les poids de run pour les trials augmentés
        we_tr = np.repeat(we_tr, aug_factor + 1)
        # poids final = poids augmentation × poids run
        wtr = w_aug * we_tr
    else:
        # sans augmentation : pénaliser les trials bruités
        wtr = (np.where(Xtr.var(axis=(1, 2)) > 1e-4, 1.0, 0.2).astype(np.float32)
               * we_tr)

    # ── 4. NORMALISATION Z-SCORE ──
    mu = sd = None
    if do_zscore:
        mu, sd = fit_channel_zscore(Xtr)
        Xtr = apply_channel_zscore(Xtr, mu, sd)
        Xva = apply_channel_zscore(Xva, mu, sd)

    # ── 5. INITIALISATION DU MODÈLE ──
    # n_ch = 16 car MU + BETA concaténés sur l'axe des canaux
    model   = EEGNet(n_ch=Xtr.shape[1], n_classes=n_classes, F1=F1, D=D, F2=F2,
                     kernel_len=kernel_len, dropout=dropout).to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    # loss pondérée : reduction='none' pour multiplier par les poids après
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    dl_tr   = DataLoader(WeightedDataset(Xtr, ytr, wtr),
                         batch_size=batch_size, shuffle=True)

    # ── 6. BOUCLE D'ENTRAÎNEMENT ──
    best_bal      = -1.0
    best_state    = None
    patience_left = patience

    for _ in range(epochs):
        # ── Phase train avec loss pondérée ──
        model.train()
        for xb, yb, wb in dl_tr:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            opt.zero_grad(set_to_none=True)
            # loss pondérée : les bons trials comptent plus
            (loss_fn(model(xb), yb) * wb).mean().backward()
            opt.step()

        # ── Phase validation ──
        with torch.no_grad():
            pred = torch.argmax(
                predict_logits(model, Xva, device), dim=1
            ).numpy()
        bal = balanced_accuracy(confusion_matrix(yva, pred, n_classes))

        # ── Early stopping ──
        if bal > best_bal + 1e-6:
            best_bal   = bal
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    # ── 7. RESTAURATION DU MEILLEUR MODÈLE ──
    if best_state:
        model.load_state_dict(best_state)

    return model, mu, sd


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                       PRÉDICTION PIPELINE 4 CLASSES                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@torch.no_grad()
def pipeline_predict_4class(gate, mu_g, sd_g,
                             expert, mu_e, sd_e,
                             Xg, Xe, idle_mask, move_mask,
                             device, thr_move,
                             gate_z, expert_z,
                             smooth_k=5):
    """
    Effectue la prédiction complète du pipeline hiérarchique 4 classes.

    Flux du pipeline :
        Pour CHAQUE trial :
            1. Le gate analyse le signal (bande MU, move1+move4)
            2. Si le trial est MOVE → l'expert prédit RIGHT/LEFT/WALK
            3. Si le trial est IDLE → prédiction finale = IDLE (classe 3)

    Smoothing temporel :
        Les prédictions de l'expert sont lissées avec une moyenne glissante
        centrée de taille smooth_k=5. Cela réduit le bruit trial-à-trial
        en exploitant la continuité temporelle des runs EEG.

        Pourquoi ça marche en EEG :
            Dans un run, les trials consécutifs du même type d'imagerie
            tendent à produire des patterns similaires. Le smoothing
            exploite cette cohérence locale pour stabiliser les prédictions.

    Note sur thr_move :
        Le paramètre thr_move est défini dans la signature mais n'est
        PAS utilisé dans cette implémentation — tous les trials MOVE
        (selon move_mask) sont passés à l'expert sans seuillage.
        Le seuillage était une feature legacy.

    Args:
        gate, expert   : modèles EEGNet entraînés.
        mu_g/sd_g      : stats normalisation gate.
        mu_e/sd_e      : stats normalisation expert.
        Xg             : entrée gate.
        Xe             : entrée expert.
        idle_mask      : True pour les trials IDLE.
        move_mask      : True pour les trials MOVE.
        device         : 'cpu' ou 'cuda'.
        thr_move       : seuil gate (non utilisé actuellement).
        gate_z/expert_z : booléens — appliquer z-score ou non.
        smooth_k       : taille fenêtre lissage temporel (défaut 5).

    Returns:
        y_pred : prédictions 4 classes (0=RIGHT,1=LEFT,2=WALK,3=IDLE).
    """
    # ── 1. NORMALISATION ──
    Xg_n = (apply_channel_zscore(Xg, mu_g, sd_g)
            if (gate_z and mu_g is not None) else Xg)
    Xe_n = (apply_channel_zscore(Xe, mu_e, sd_e)
            if (expert_z and mu_e is not None) else Xe)

    # ── 2. INITIALISATION : tout IDLE par défaut ──
    # Si le gate ou l'expert ne prédit rien sur un trial, il reste IDLE
    y_pred = np.full(Xg.shape[0], 3, dtype=np.int64)  # 3 = IDLE

    # ── 3. GATE (non utilisé pour filtrer ici — move_mask est utilisé) ──
    logits_g = predict_logits(gate, Xg_n, device)
    p_move   = torch.softmax(logits_g, dim=1)[:, 1].cpu().numpy()
    # Note : p_move calculé mais thr_move non appliqué dans cette version

    # ── 4. EXPERT sur les trials MOVE ──
    idx = np.where(move_mask)[0]  # indices des trials MOVE dans Xg/Xe
    if len(idx) > 0:
        logits_e = predict_logits(expert, Xe_n[idx], device)
        probs    = torch.softmax(logits_e, dim=1).cpu().numpy()

        # ── 5. SMOOTHING TEMPOREL ──
        if smooth_k > 1:
            half         = smooth_k // 2
            probs_smooth = probs.copy()
            for i in range(half, len(probs) - half):
                # moyenne des probabilités sur smooth_k trials consécutifs
                probs_smooth[i] = probs[i - half:i + half + 1].mean(axis=0)
            probs = probs_smooth

        # ── 6. ARGMAX → classe finale ──
        y_pred[idx] = np.argmax(probs, axis=1).astype(np.int64)

    return y_pred


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                    OPTIMISATION DU SEUIL thr_move                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def find_best_thr_on_train(gate, expert,
                           Xg_train_val, Xe_train_val,
                           y_train_val,
                           idle_mask_val, move_mask_val,
                           device, gate_z, expert_z,
                           mu_g, sd_g, mu_e, sd_e):
    """
    Optimise le seuil de décision du gate sur des données d'entraînement.

    Pourquoi optimiser le seuil :
        Le seuil thr_move contrôle la sensibilité du gate — un seuil bas
        (0.20) détecte presque tout comme MOVE, un seuil haut (0.50)
        est très conservateur. Le seuil optimal dépend du sujet et
        de la session.

    Anti-leakage CRUCIAL :
        Le seuil est optimisé sur un run d'ENTRAÎNEMENT (pas de test).
        Utiliser le run de test pour optimiser le seuil serait du data
        leakage et gonflerait artificiellement les performances.

        En pratique : on utilise le run (test_idx + 1) % n_runs
        (le run "suivant" dans le LORO) ou le dernier run train
        (en mode Train/Test).

    Args:
        gate, expert       : modèles entraînés.
        Xg/Xe/y_train_val  : données du run de validation sur l'entraînement.
        idle/move_mask_val : masques IDLE/MOVE pour ce run.
        [params normalisation et device]

    Returns:
        best_thr : float, seuil optimal parmi [0.20, 0.25, ..., 0.50].
    """
    thr_list  = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    best_score = -1
    best_thr   = 0.30  # valeur par défaut raisonnable

    for thr in thr_list:
        y_pred = pipeline_predict_4class(
            gate, mu_g, sd_g, expert, mu_e, sd_e,
            Xg_train_val, Xe_train_val,
            idle_mask_val, move_mask_val,
            device, thr, gate_z, expert_z
        )
        bal = balanced_accuracy(confusion_matrix(y_train_val, y_pred, 4))
        if bal > best_score:
            best_score = bal
            best_thr   = thr

    return best_thr


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                       CHARGEMENT ET NORMALISATION DES RUNS                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def load_runs(npz_path):
    """
    Charge les runs EEG depuis un fichier .npz.

    Structure attendue du .npz :
        runs : numpy array d'objets, chaque élément est un dict avec :
            - y         : labels des trials (entiers 1,2,3)
            - X_move1   : signal fenêtre move1 (n_trials, n_ch, n_time)
            - X_move4   : signal fenêtre move4
            - X_nomove1 : signal fenêtre repos 1
            - X_nomove2 : signal fenêtre repos 2
            - [autres fenêtres move2, move3, move5, move6]
            - fs_used   : fréquence d'échantillonnage utilisée
            - band      : [freq_low, freq_high]

    Args:
        npz_path : chemin vers le fichier .npz.

    Returns:
        list[dict] : liste de runs.
    """
    d = np.load(npz_path, allow_pickle=True)
    if "runs" not in d:
        raise RuntimeError(f"Clé 'runs' introuvable dans {npz_path}. "
                          f"Clés disponibles : {list(d.keys())}")
    return normalize_runs_array(d["runs"])


def zscore_runs_independent(runs):
    """
    Normalise chaque run EEG indépendamment (Z-score par run).

    Philosophie anti-leakage :
        Dans un paradigme multi-jours ou multi-sessions, les distributions
        du signal EEG changent entre les sessions (drift d'amplitudes,
        changement de qualité de contact, etc.). Normaliser chaque run
        indépendamment garantit qu'aucune information d'une session ne
        "fuite" vers une autre session via la normalisation.

        Sans normalisation indépendante :
            Si on normalise tout le dataset ensemble, les runs avec de
            grands amplitudes "pollueraient" les statistiques globales,
            rendant les petits runs quasi-nuls après normalisation.

    Application :
        Uniquement sur les clés X_* (signaux EEG).
        Les métadonnées (y, fs, band, etc.) sont conservées intactes.

    Args:
        runs : list[dict] de runs EEG bruts.

    Returns:
        list[dict] : runs normalisés, même structure que l'entrée.
    """
    runs_out = []
    for run in runs:
        run     = as_run_dict(run)
        new_run = {}
        for k, v in run.items():
            if k.startswith("X_"):
                # normalisation indépendante pour chaque fenêtre temporelle
                X        = np.array(v).astype(np.float32)
                mu, sd   = fit_channel_zscore(X)
                new_run[k] = apply_channel_zscore(X, mu, sd)
            else:
                # conservation des labels et métadonnées
                new_run[k] = v
        runs_out.append(new_run)
    return runs_out


def build_train_arrays_from_runs(runs_mu, runs_b, idle_w, move_w_gate,
                                 move_w_expert, stack_mode_gate,
                                 stack_mode_expert, recalibration_runs=0):
    """
    Construit les arrays d'entraînement en fusionnant tous les runs.

    Pondération des runs récents (recalibration) :
        Les derniers runs d'entraînement (recalibration_runs) reçoivent
        un poids de 1.5 au lieu de 1.0. En BCI adaptatif, les données
        récentes sont plus représentatives des patterns actuels du sujet
        (le signal EEG dérive lentement au cours d'une session).

    Args:
        runs_mu/runs_b         : runs bandes MU et BETA.
        idle_w, move_w_gate    : fenêtres pour le gate.
        move_w_expert          : fenêtres pour l'expert.
        stack_mode_gate/expert : modes d'empilement.
        recalibration_runs     : nombre de runs récents à surpondérer.

    Returns:
        Xg_train, yg_train : données et labels pour le gate.
        Xe_train, ye_train : données et labels pour l'expert.
        we_train           : poids par trial expert.
    """
    Xg_list, yg_list = [], []
    Xe_list, ye_list = [], []
    we_list          = []
    n_runs = len(runs_mu)
    # seuil à partir duquel les runs sont "récents" (à surpondérer)
    n_base = n_runs - recalibration_runs

    for i in range(n_runs):
        # construction datasets gate et expert pour ce run
        Xg_i, yg_i = build_gate_idle_vs_move(
            runs_mu[i], idle_w, move_w_gate, stack_mode_gate)
        Xe_i, ye_i = build_expert_rl(
            runs_mu[i], runs_b[i], move_w_expert, stack_mode_expert)

        # poids du run : 1.5 pour les runs récents de recalibration, 1.0 sinon
        w = 1.5 if i >= n_base else 1.0

        Xg_list.append(Xg_i); yg_list.append(yg_i)
        Xe_list.append(Xe_i); ye_list.append(ye_i)
        we_list.append(np.full(len(ye_i), w, dtype=np.float32))

    return (
        np.concatenate(Xg_list), np.concatenate(yg_list),
        np.concatenate(Xe_list), np.concatenate(ye_list),
        np.concatenate(we_list)
    )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           MODE LORO                                          ║
# ╠══════════════════════════════════════════════════════════════════════════════╣
# ║  Leave-One-Run-Out : pour configs mono-jour (Jour 1, 2, 3 seuls)            ║
# ║  Entraîne sur n-1 runs, teste sur le run restant, répète n fois             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def run_loro_mode(args, device):
    """
    Lance l'évaluation en mode LORO (Leave-One-Run-Out).

    Pourquoi LORO pour les configs mono-jour :
        Un jour contient ~8 runs de ~30 trials. Le LORO teste sur chaque
        run à tour de rôle en s'entraînant sur les 7 autres. C'est une
        validation robuste qui simule la situation réelle où on entraîne
        sur quelques runs puis on évalue sur les suivants.

        Contrairement à la 10-fold CV de RLDA, le LORO respecte la
        structure temporelle des runs (pas de mélange entre runs).

    Résultats reportés :
        - accuracy par run (% de trials correctement classifiés)
        - balanced_accuracy par run (métrique principale)
        - matrice de confusion par run
        - mean_acc et mean_bal_acc globaux
    """
    # chargement des fichiers .npz (MU et BETA)
    runs_mu = normalize_runs_array(
        np.load(args.npz_mu, allow_pickle=True)["runs"])
    runs_b  = normalize_runs_array(
        np.load(args.npz_beta, allow_pickle=True)["runs"])
    assert len(runs_mu) == len(runs_b), \
        f"Nombre de runs MU ({len(runs_mu)}) ≠ BETA ({len(runs_b)})"
    n_runs = len(runs_mu)

    idle_w      = parse_windows(args.idle_windows)
    move_w_gate = parse_windows(args.move_windows)
    move_w_exp  = parse_windows(args.move_windows_expert)

    print(f"Loaded MU  : {args.npz_mu} | nb_runs={n_runs}")
    print(f"Loaded BETA: {args.npz_beta}")
    print(f"idle_windows={idle_w}")
    print(f"move_windows gate={move_w_gate} | expert={move_w_exp}")
    print(f"expert aug_factor={args.expert_aug_factor}")
    print()

    labels4 = ["RIGHT", "LEFT", "WALK", "IDLE"]
    per_run, accs, bals = [], [], []

    for test_idx in range(n_runs):
        # construction des données de test pour ce run
        Xg_te, Xe_te, y_true, idle_mask, move_mask = build_test_eval_4class(
            runs_mu[test_idx], runs_b[test_idx],
            idle_w, move_w_gate, move_w_exp,
            args.stack_mode_gate, args.stack_mode_expert)

        # construction des données d'entraînement (tous les autres runs)
        Xg_tr_list, yg_tr_list = [], []
        Xe_tr_list, ye_tr_list = [], []
        we_list = []

        for i in range(n_runs):
            if i == test_idx:
                continue  # exclure le run de test
            Xg_i, yg_i = build_gate_idle_vs_move(
                runs_mu[i], idle_w, move_w_gate, args.stack_mode_gate)
            Xe_i, ye_i = build_expert_rl(
                runs_mu[i], runs_b[i], move_w_exp, args.stack_mode_expert)
            Xg_tr_list.append(Xg_i); yg_tr_list.append(yg_i)
            Xe_tr_list.append(Xe_i); ye_tr_list.append(ye_i)
            we_list.append(np.ones(len(ye_i), dtype=np.float32))

        Xg_train = np.concatenate(Xg_tr_list)
        yg_train = np.concatenate(yg_tr_list)
        Xe_train = np.concatenate(Xe_tr_list)
        ye_train = np.concatenate(ye_tr_list)
        we_train = np.concatenate(we_list)

        # entraînement gate
        print(f"  [Run {test_idx}] Training gate...")
        gate, mu_g, sd_g = train_gate(
            Xg_train, yg_train, device, n_classes=2,
            val_split=args.val_split,
            seed_split=args.seed + 1000 + test_idx,
            do_zscore=args.gate_train_zscore,
            kernel_len=args.kernel_gate,
            dropout=args.dropout_gate, lr=args.lr, wd=args.wd,
            batch_size=args.batch_size, epochs=args.epochs,
            patience=args.patience, F1=args.F1, D=args.D, F2=args.F2)

        # entraînement expert
        print(f"  [Run {test_idx}] Training expert...")
        expert, mu_e, sd_e = train_expert(
            Xe_train, ye_train, device, n_classes=3,
            val_split=args.val_split,
            seed_split=args.seed + 3000 + test_idx,
            do_zscore=args.expert_train_zscore,
            kernel_len=args.kernel_expert,
            dropout=args.dropout_expert, lr=args.lr, wd=args.wd,
            batch_size=args.batch_size, epochs=args.epochs,
            patience=args.patience, F1=args.F1, D=args.D, F2=args.F2,
            aug_factor=args.expert_aug_factor, we_all=we_train)

        # optimisation du seuil sur un run de TRAIN (pas le run de test)
        # → run (test_idx + 1) % n_runs = le "run suivant" dans la rotation
        val_run_idx = (test_idx + 1) % n_runs
        Xg_val, Xe_val, y_val, idle_val, move_val = build_test_eval_4class(
            runs_mu[val_run_idx], runs_b[val_run_idx],
            idle_w, move_w_gate, move_w_exp,
            args.stack_mode_gate, args.stack_mode_expert)

        best_thr = find_best_thr_on_train(
            gate, expert, Xg_val, Xe_val, y_val,
            idle_val, move_val, device,
            args.gate_train_zscore, args.expert_train_zscore,
            mu_g, sd_g, mu_e, sd_e)
        print(f"    [THR] thr={best_thr:.2f} (optimisé sur run train)")

        # prédiction sur le run de test
        y_pred = pipeline_predict_4class(
            gate, mu_g, sd_g, expert, mu_e, sd_e,
            Xg_te, Xe_te, idle_mask, move_mask,
            device, best_thr,
            args.gate_train_zscore, args.expert_train_zscore,
            smooth_k=args.smooth_k)

        cm  = confusion_matrix(y_true, y_pred, 4)
        acc = float((y_true == y_pred).mean())
        bal = balanced_accuracy(cm)
        per_run.append((acc, bal, cm, len(y_true)))
        accs.append(acc)
        bals.append(bal)

    # affichage des résultats
    print("==== PER-RUN RESULTS (LORO, 4-class) ====")
    for run_idx, (acc, bal, cm, n) in enumerate(per_run):
        print(f"[Run {run_idx}] acc={acc:.3f} | bal_acc={bal:.3f} | n={n}")
        print_cm(cm, labels4)
        print()

    print("==== GLOBAL SUMMARY ====")
    print(f"mean_acc={np.mean(accs):.6f} | mean_bal_acc={np.mean(bals):.6f}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                        MODE TRAIN/TEST                                       ║
# ╠══════════════════════════════════════════════════════════════════════════════╣
# ║  Pour configs multi-jours : entraîne sur jour(s) A, teste sur jour(s) B    ║
# ║  Ex: entraîne sur Jour 1, teste sur Jour 2 (config "1+2")                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def run_train_test_mode(args, device):
    """
    Lance l'évaluation en mode Train/Test inter-sessions.

    Principe :
        - Train : données du/des jour(s) d'entraînement
        - Test  : données du jour à évaluer

    Recalibration optionnelle :
        Si --recalibration_runs=k, les k premiers runs du jour de test
        sont ajoutés à l'entraînement (avec poids 1.5). Cela simule
        une adaptation du modèle au début d'une nouvelle session.

        Exemple avec k=2 :
            Train : Jour 1 complet + 2 premiers runs Jour 2
            Test  : runs 3 à 8 du Jour 2

    Fine-tuning si recalibration :
        Après l'entraînement général, l'expert est fine-tuné 5 epochs
        sur les runs de recalibration avec lr/10 pour s'adapter à la
        session courante sans oublier ce qui a été appris (catastrophic
        forgetting limité par le lr faible et le peu d'epochs).
    """
    # chargement et normalisation indépendante par run
    runs_mu_train_base = zscore_runs_independent(load_runs(args.npz_mu_train))
    runs_b_train_base  = zscore_runs_independent(load_runs(args.npz_beta_train))
    runs_mu_test_all   = zscore_runs_independent(load_runs(args.npz_mu_test))
    runs_b_test_all    = zscore_runs_independent(load_runs(args.npz_beta_test))

    k = args.recalibration_runs
    if k < 0:
        raise RuntimeError("--recalibration_runs doit être >= 0")
    if k >= len(runs_mu_test_all):
        raise RuntimeError(
            f"--recalibration_runs={k} >= nb_runs_test={len(runs_mu_test_all)}")

    # fusion train de base + runs de recalibration
    runs_mu_train = list(runs_mu_train_base) + list(runs_mu_test_all[:k])
    runs_b_train  = list(runs_b_train_base)  + list(runs_b_test_all[:k])
    # runs de test = runs du jour de test APRÈS la recalibration
    runs_mu_test  = list(runs_mu_test_all[k:])
    runs_b_test   = list(runs_b_test_all[k:])

    idle_w      = parse_windows(args.idle_windows)
    move_w_gate = parse_windows(args.move_windows)
    move_w_exp  = parse_windows(args.move_windows_expert)

    print(f"Train runs={len(runs_mu_train)} | Test runs={len(runs_mu_test)}")
    print(f"move_windows gate={move_w_gate} | expert={move_w_exp}")
    print(f"expert aug_factor={args.expert_aug_factor}")
    print()

    # construction des données d'entraînement
    Xg_train, yg_train, Xe_train, ye_train, we_train = build_train_arrays_from_runs(
        runs_mu_train, runs_b_train, idle_w, move_w_gate, move_w_exp,
        args.stack_mode_gate, args.stack_mode_expert,
        recalibration_runs=args.recalibration_runs)

    labels4 = ["RIGHT", "LEFT", "WALK", "IDLE"]

    # entraînement gate
    print("  Training gate...")
    gate, mu_g, sd_g = train_gate(
        Xg_train, yg_train, device, n_classes=2,
        val_split=args.val_split, seed_split=args.seed + 1000,
        do_zscore=args.gate_train_zscore, kernel_len=args.kernel_gate,
        dropout=args.dropout_gate, lr=args.lr, wd=args.wd,
        batch_size=args.batch_size, epochs=args.epochs,
        patience=args.patience, F1=args.F1, D=args.D, F2=args.F2)

    # entraînement expert
    print("  Training expert...")
    expert, mu_e, sd_e = train_expert(
        Xe_train, ye_train, device, n_classes=3,
        val_split=args.val_split, seed_split=args.seed + 3000,
        do_zscore=args.expert_train_zscore, kernel_len=args.kernel_expert,
        dropout=args.dropout_expert, lr=args.lr, wd=args.wd,
        batch_size=args.batch_size, epochs=args.epochs,
        patience=args.patience, F1=args.F1, D=args.D, F2=args.F2,
        aug_factor=args.expert_aug_factor, we_all=we_train)

    # fine-tuning optionnel sur les runs de recalibration
    if args.recalibration_runs > 0:
        print("  [FT] fine-tuning expert sur runs de recalibration...")
        Xe_ft_list, ye_ft_list = [], []
        for i in range(args.recalibration_runs):
            Xe_i, ye_i = build_expert_rl(
                runs_mu_train[-args.recalibration_runs + i],
                runs_b_train[-args.recalibration_runs + i],
                move_w_exp, args.stack_mode_expert)
            Xe_ft_list.append(Xe_i)
            ye_ft_list.append(ye_i)
        Xe_ft = np.concatenate(Xe_ft_list)
        ye_ft = np.concatenate(ye_ft_list)
        expert.train()
        # lr/10 pour fine-tuning conservateur (éviter catastrophic forgetting)
        opt_ft = torch.optim.Adam(expert.parameters(), lr=args.lr * 0.1)
        for _ in range(5):
            xb = torch.from_numpy(Xe_ft).float().to(device)
            yb = torch.from_numpy(ye_ft).long().to(device)
            opt_ft.zero_grad()
            nn.CrossEntropyLoss()(expert(xb), yb).backward()
            opt_ft.step()

    # optimisation du seuil sur le dernier run de TRAIN
    Xg_val, Xe_val, y_val, idle_val, move_val = build_test_eval_4class(
        runs_mu_train[-1], runs_b_train[-1],
        idle_w, move_w_gate, move_w_exp,
        args.stack_mode_gate, args.stack_mode_expert)

    best_thr = find_best_thr_on_train(
        gate, expert, Xg_val, Xe_val, y_val,
        idle_val, move_val, device,
        args.gate_train_zscore, args.expert_train_zscore,
        mu_g, sd_g, mu_e, sd_e)
    print(f"[THR] thr={best_thr:.2f} (optimisé sur dernier run train)")

    # évaluation sur les runs de test
    per_run, accs, bals = [], [], []
    for test_idx in range(len(runs_mu_test)):
        Xg_te, Xe_te, y_true, idle_mask, move_mask = build_test_eval_4class(
            runs_mu_test[test_idx], runs_b_test[test_idx],
            idle_w, move_w_gate, move_w_exp,
            args.stack_mode_gate, args.stack_mode_expert)

        y_pred = pipeline_predict_4class(
            gate, mu_g, sd_g, expert, mu_e, sd_e,
            Xg_te, Xe_te, idle_mask, move_mask,
            device, best_thr,
            args.gate_train_zscore, args.expert_train_zscore,
            smooth_k=args.smooth_k)

        cm  = confusion_matrix(y_true, y_pred, 4)
        acc = float((y_true == y_pred).mean())
        bal = balanced_accuracy(cm)
        per_run.append((acc, bal, cm, len(y_true)))
        accs.append(acc)
        bals.append(bal)

    print("==== PER-RUN RESULTS ====")
    for run_idx, (acc, bal, cm, n) in enumerate(per_run):
        print(f"[Run {run_idx}] acc={acc:.3f} | bal_acc={bal:.3f} | n={n}")
        print_cm(cm, labels4)
        print()

    print("==== GLOBAL SUMMARY ====")
    print(f"mean_acc={np.mean(accs):.6f} | mean_bal_acc={np.mean(bals):.6f}")




# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║              FONCTIONS AVEC COURBES D'ENTRAÎNEMENT                          ║
# ║  À ajouter à la fin de eegnet_4class.py pour générer les figures            ║
# ║  du Chapitre 5 (courbes train/val + epoch d'early stopping)                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def train_gate_with_curves(X_all, y_all, device, *, n_classes, val_split,
                            seed_split, do_zscore, kernel_len, dropout, lr,
                            wd, batch_size, epochs, patience, F1, D, F2,
                            **kwargs):
    """
    Version de train_gate qui retourne aussi les courbes d'entraînement.

    Identique à train_gate mais enregistre la balanced accuracy à chaque
    epoch sur train ET validation, ainsi que l'epoch d'early stopping.

    Returns:
        model    : EEGNet gate entraîné
        mu, sd   : statistiques de normalisation
        tr_bals  : liste des balanced accuracy train par epoch
        val_bals : liste des balanced accuracy validation par epoch
        early_ep : epoch d'early stopping
    """
    tr_idx, va_idx = stratified_split_indices(y_all, val_split, seed_split)
    if len(va_idx) == 0:
        rng = np.random.RandomState(seed_split)
        idx = np.arange(len(y_all)); rng.shuffle(idx)
        n_val = max(1, int(0.05 * len(idx)))
        va_idx = idx[:n_val]; tr_idx = idx[n_val:]

    Xtr, ytr = X_all[tr_idx], y_all[tr_idx]
    Xva, yva = X_all[va_idx], y_all[va_idx]

    mu = sd = None
    if do_zscore:
        mu, sd = fit_channel_zscore(Xtr)
        Xtr = apply_channel_zscore(Xtr, mu, sd)
        Xva = apply_channel_zscore(Xva, mu, sd)

    model   = EEGNet(n_ch=Xtr.shape[1], n_classes=n_classes, F1=F1, D=D,
                     F2=F2, kernel_len=kernel_len, dropout=dropout).to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()
    dl_tr   = DataLoader(SimpleDataset(Xtr, ytr),
                         batch_size=batch_size, shuffle=True)

    best_bal      = -1.0
    best_state    = None
    patience_left = patience
    early_ep      = epochs
    tr_bals, val_bals = [], []

    for ep in range(epochs):
        # ── Phase train ──
        model.train()
        tr_preds, tr_true = [], []
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss_fn(out, yb).backward()
            opt.step()
            tr_preds.extend(out.argmax(1).cpu().numpy())
            tr_true.extend(yb.cpu().numpy())

        tr_bal = balanced_accuracy(
            confusion_matrix(np.array(tr_true), np.array(tr_preds), n_classes))
        tr_bals.append(tr_bal)

        # ── Phase validation ──
        with torch.no_grad():
            pred = torch.argmax(
                predict_logits(model, Xva, device), dim=1).numpy()
        val_bal = balanced_accuracy(confusion_matrix(yva, pred, n_classes))
        val_bals.append(val_bal)

        # ── Early stopping ──
        if val_bal > best_bal + 1e-6:
            best_bal      = val_bal
            best_state    = {k: v.detach().cpu().clone()
                             for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                early_ep = ep + 1
                break

    if best_state:
        model.load_state_dict(best_state)

    return model, mu, sd, tr_bals, val_bals, early_ep


def train_expert_with_curves(X_all, y_all, device, *, n_classes, val_split,
                              seed_split, do_zscore, kernel_len, dropout, lr,
                              wd, batch_size, epochs, patience, F1, D, F2,
                              aug_factor=9, we_all=None, **kwargs):
    """
    Version de train_expert qui retourne aussi les courbes d'entraînement.

    Identique à train_expert mais enregistre la balanced accuracy à chaque
    epoch sur train ET validation, ainsi que l'epoch d'early stopping.

    Returns:
        model    : EEGNet expert entraîné
        mu, sd   : statistiques de normalisation
        tr_bals  : liste des balanced accuracy train par epoch
        val_bals : liste des balanced accuracy validation par epoch
        early_ep : epoch d'early stopping
    """
    tr_idx, va_idx = stratified_split_indices(y_all, val_split, seed_split)
    if len(va_idx) == 0:
        rng = np.random.RandomState(seed_split)
        idx = np.arange(len(y_all)); rng.shuffle(idx)
        n_val = max(1, int(0.05 * len(idx)))
        va_idx = idx[:n_val]; tr_idx = idx[n_val:]

    Xtr, ytr = X_all[tr_idx], y_all[tr_idx]
    Xva, yva = X_all[va_idx], y_all[va_idx]

    we_tr = (we_all[tr_idx] if we_all is not None
             else np.ones(len(ytr), dtype=np.float32))

    if aug_factor > 0:
        n_before = len(ytr)
        Xtr, ytr, w_aug = augment_expert_data(
            Xtr, ytr, aug_factor=aug_factor, seed=seed_split + 999)
        print(f"    [EXPERT_AUG] {n_before} → {len(ytr)} trials (x{aug_factor+1})")
        we_tr = np.repeat(we_tr, aug_factor + 1)
        wtr = w_aug * we_tr
    else:
        wtr = (np.where(Xtr.var(axis=(1, 2)) > 1e-4, 1.0, 0.2).astype(np.float32)
               * we_tr)

    mu = sd = None
    if do_zscore:
        mu, sd = fit_channel_zscore(Xtr)
        Xtr = apply_channel_zscore(Xtr, mu, sd)
        Xva = apply_channel_zscore(Xva, mu, sd)

    model   = EEGNet(n_ch=Xtr.shape[1], n_classes=n_classes, F1=F1, D=D,
                     F2=F2, kernel_len=kernel_len, dropout=dropout).to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    dl_tr   = DataLoader(WeightedDataset(Xtr, ytr, wtr),
                         batch_size=batch_size, shuffle=True)

    best_bal      = -1.0
    best_state    = None
    patience_left = patience
    early_ep      = epochs
    tr_bals, val_bals = [], []

    for ep in range(epochs):
        # ── Phase train ──
        model.train()
        tr_preds, tr_true = [], []
        for xb, yb, wb in dl_tr:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            (loss_fn(out, yb) * wb).mean().backward()
            opt.step()
            tr_preds.extend(out.argmax(1).cpu().numpy())
            tr_true.extend(yb.cpu().numpy())

        tr_bal = balanced_accuracy(
            confusion_matrix(np.array(tr_true), np.array(tr_preds), n_classes))
        tr_bals.append(tr_bal)

        # ── Phase validation ──
        with torch.no_grad():
            pred = torch.argmax(
                predict_logits(model, Xva, device), dim=1).numpy()
        val_bal = balanced_accuracy(confusion_matrix(yva, pred, n_classes))
        val_bals.append(val_bal)

        # ── Early stopping ──
        if val_bal > best_bal + 1e-6:
            best_bal      = val_bal
            best_state    = {k: v.detach().cpu().clone()
                             for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                early_ep = ep + 1
                break

    if best_state:
        model.load_state_dict(best_state)

    return model, mu, sd, tr_bals, val_bals, early_ep

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              POINT D'ENTRÉE                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main():
    """
    Point d'entrée principal du script.

    Détection automatique du mode :
        - Si --npz_mu et --npz_beta fournis → mode LORO
        - Si --npz_mu_train et --npz_mu_test fournis → mode Train/Test
        - Les deux peuvent être fournis (Train/Test a la priorité)
    """
    ap = argparse.ArgumentParser(
        description="Pipeline EEGNet 4 classes pour BCI imagerie motrice",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ── Entrées données ──
    ap.add_argument("--npz_mu",   default=None,
                    help="[LORO] .npz bande MU du jour à évaluer")
    ap.add_argument("--npz_beta", default=None,
                    help="[LORO] .npz bande BETA du jour à évaluer")
    ap.add_argument("--npz_mu_train",   default=None,
                    help="[T/T] .npz bande MU pour l'entraînement")
    ap.add_argument("--npz_beta_train", default=None,
                    help="[T/T] .npz bande BETA pour l'entraînement")
    ap.add_argument("--npz_mu_test",    default=None,
                    help="[T/T] .npz bande MU pour le test")
    ap.add_argument("--npz_beta_test",  default=None,
                    help="[T/T] .npz bande BETA pour le test")
    ap.add_argument("--recalibration_runs", type=int, default=0,
                    help="Nombre de runs de test utilisés pour recalibration")

    # ── Sélection des fenêtres temporelles ──
    ap.add_argument("--idle_windows",        default="nomove1,nomove2",
                    help="Fenêtres repos pour le gate")
    ap.add_argument("--move_windows",        default="move1,move4",
                    help="Fenêtres mouvement pour le gate")
    ap.add_argument("--move_windows_expert", default="move1",
                    help="Fenêtres mouvement pour l'expert")

    # ── Mode d'empilement des fenêtres ──
    ap.add_argument("--stack_mode_gate",   choices=["time","trials"], default="time",
                    help="'time'=concat temporelle, 'trials'=concat trials")
    ap.add_argument("--stack_mode_expert", choices=["time","trials"], default="time")

    # ── Options de prétraitement ──
    ap.add_argument("--gate_train_zscore",   action="store_true",
                    help="Appliquer z-score sur les données du gate")
    ap.add_argument("--expert_train_zscore", action="store_true",
                    help="Appliquer z-score sur les données de l'expert")

    # ── Hyperparamètres d'entraînement ──
    ap.add_argument("--val_split",  type=float, default=0.2,
                    help="Fraction des données pour la validation")
    ap.add_argument("--smooth_k",   type=int,   default=5,
                    help="Taille fenêtre lissage temporel des prédictions")
    ap.add_argument("--lr",         type=float, default=1e-3,
                    help="Learning rate Adam")
    ap.add_argument("--wd",         type=float, default=1e-4,
                    help="Weight decay Adam")
    ap.add_argument("--batch_size", type=int,   default=64)
    ap.add_argument("--epochs",     type=int,   default=200,
                    help="Nombre maximum d'epochs")
    ap.add_argument("--patience",   type=int,   default=40,
                    help="Early stopping patience")

    # ── Architecture EEGNet ──
    ap.add_argument("--F1", type=int, default=8,
                    help="Nombre de filtres temporels")
    ap.add_argument("--D",  type=int, default=2,
                    help="Profondeur spatiale (F1*D canaux après bloc spatial)")
    ap.add_argument("--F2", type=int, default=16,
                    help="Nombre de filtres séparables")
    ap.add_argument("--dropout_gate",      type=float, default=0.25)
    ap.add_argument("--dropout_expert",    type=float, default=0.25)
    ap.add_argument("--kernel_gate",       type=int,   default=128,
                    help="Longueur filtre temporel gate (samples)")
    ap.add_argument("--kernel_expert",     type=int,   default=192,
                    help="Longueur filtre temporel expert (samples)")
    ap.add_argument("--expert_aug_factor", type=int,   default=9,
                    help="Facteur augmentation expert (aug_factor+1 = total multiplier)")

    # ── Runtime ──
    ap.add_argument("--seed",   type=int, default=0,
                    help="Graine aléatoire pour reproductibilité")
    ap.add_argument("--device", type=str, default="cpu",
                    help="'cpu' ou 'cuda' pour GPU")

    args = ap.parse_args()

    # initialisation de la reproductibilité
    set_seed(args.seed)
    device = torch.device(args.device)

    # détection du mode d'exécution
    has_loro      = args.npz_mu is not None
    has_traintest = (args.npz_mu_train is not None
                     and args.npz_mu_test is not None)

    if not has_loro and not has_traintest:
        raise RuntimeError(
            "Fournis soit --npz_mu + --npz_beta (mode LORO) "
            "soit --npz_mu_train + --npz_mu_test (mode Train/Test)"
        )

    # lancement du pipeline
    if has_traintest:
        run_train_test_mode(args, device)
    else:
        run_loro_mode(args, device)


if __name__ == "__main__":
    main()