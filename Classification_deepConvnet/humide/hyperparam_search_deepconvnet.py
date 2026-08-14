#!/usr/bin/env python3
"""
hyperparam_search_deepconvnet.py — Recherche d'hyperparamètres DeepConvNet
===========================================================================
Développé dans le cadre du projet BCI-EEG ÉTS (2026)
Auteure : Emma

OBJECTIF :
    Ce script identifie la configuration optimale des hyperparamètres
    de DeepConvNet pour les données à électrodes humides, en testant
    une variable à la fois par rapport à une configuration de base.
    Parallèle à hyperparam_search_fast.py pour EEGNet.

PROTOCOLE D'ÉVALUATION :
    LORO partiel : 3 folds (runs 1, 5 et 10) sur 3 sujets représentatifs
    (Subj04, Subj05, Subj07), avec 60 epochs maximum et patience 15.
    La configuration retenue est celle donnant la meilleure balanced
    accuracy moyenne sur les 3 sujets × 3 folds = 9 entraînements.

CONFIGURATIONS TESTÉES (une variable à la fois) :
    - Configuration de base : dropout=0.5, lr=5e-4, bande=8-30Hz
    - Dropout : 0.25, 0.5 (base)
    - Learning rate : 1e-3, 5e-4 (base), 1e-4
    - Bande de fréquence : 8-30Hz (base), 8-13Hz (mu), 13-30Hz (bêta)

USAGE :
    python hyperparam_search_deepconvnet.py

RÉSULTATS :
    ./hyperparam_results_deepconvnet/results_loro.csv

DÉPENDANCES :
    pip install numpy torch scikit-learn
"""

import os
import sys
import csv
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import balanced_accuracy_score
from pathlib import Path

# ── Chemins ──────────────────────────────────────────────────────────────────
BASE_DIR   = "/Users/emma/Desktop/Code/code_pipeline"
NPZ_DAY1   = os.path.join(BASE_DIR, "npz_all_subjects_day1")
OUTPUT_DIR = os.path.join(BASE_DIR, "hyperparam_results_deepconvnet")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Sujets de référence ───────────────────────────────────────────────────────
SUBJECTS = ["Subj04", "Subj05", "Subj07"]

# ── Folds LORO partiels ───────────────────────────────────────────────────────
LORO_TEST_FOLDS = [0, 4, 9]  # runs 1, 5, 10

# ── Configuration de base ─────────────────────────────────────────────────────
BASE = {
    "dropout" : 0.5,
    "lr"      : 5e-4,
    "band"    : "8-30Hz",
}

# ── Configurations à tester ───────────────────────────────────────────────────
CONFIGS = [
    ("base",          {**BASE}),
    ("dropout-025",   {**BASE, "dropout": 0.25}),
    ("lr-1e3",        {**BASE, "lr": 1e-3}),
    ("lr-1e4",        {**BASE, "lr": 1e-4}),
    ("band-mu",       {**BASE, "band": "8-13Hz"}),
    ("band-beta",     {**BASE, "band": "13-30Hz"}),
]

# ── Hyperparamètres fixes ─────────────────────────────────────────────────────
FAST_EPOCHS  = 60
FAST_PATIENCE = 15
BATCH_SIZE   = 64
SEED         = 42

device = torch.device("cpu")


# ============================================================
# ARCHITECTURE DEEPCONVNET
# ============================================================

class DeepConvNet(nn.Module):
    """
    Architecture DeepConvNet avec dropout configurable.

    3 blocs convolutifs successifs (25 → 50 → 100 filtres),
    chacun avec convolution temporelle, activation ELU et MaxPooling.
    Le premier bloc ajoute une convolution spatiale sur les canaux EEG.

    Args:
        n_channels : nombre de canaux EEG (16 pour Kaptics, 8 pour humide)
        n_classes  : nombre de classes à prédire
        dropout    : taux de dropout entre les blocs (0.0 = désactivé)
    """
    def __init__(self, n_channels=8, n_classes=4, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            # Bloc 1 : convolution temporelle + spatiale
            nn.Conv2d(1, 25, (1, 64), padding=(0, 32)),
            nn.Conv2d(25, 25, (n_channels, 1)),
            nn.ELU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(dropout),

            # Bloc 2 : patterns de plus haut niveau
            nn.Conv2d(25, 50, (1, 64), padding=(0, 32)),
            nn.ELU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(dropout),

            # Bloc 3 : représentations abstraites
            nn.Conv2d(50, 100, (1, 64), padding=(0, 32)),
            nn.ELU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(dropout),
        )
        # Calcul automatique de la taille de sortie
        dummy  = torch.zeros(1, 1, n_channels, 250)
        n_flat = self.net(dummy).flatten(1).shape[1]
        self.fc = nn.Linear(n_flat, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.fc(self.net(x).flatten(1))


# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================

def load_npz(path):
    """
    Charge un fichier .npz et retourne la liste des runs.
    Format : clé 'runs' contenant un array de dicts.
    Chaque dict contient X_nomove1, X_move1, X_move4, etc.
    """
    data = np.load(path, allow_pickle=True)
    return list(data['runs'])


def get_npz_paths(subj, band):
    """
    Retourne les chemins des fichiers .npz selon la bande de fréquence.
    Pour band-mu : fichier band0812 comme MU et BETA (MU seule)
    Pour band-beta : fichier band1330 comme MU et BETA (BETA seule)
    Pour 8-30Hz : fichiers band0812 (MU) et band1330 (BETA)
    """
    mu   = os.path.join(NPZ_DAY1, f"{subj}_Jour1_band0812_motor8_guessB.npz")
    beta = os.path.join(NPZ_DAY1, f"{subj}_Jour1_band1330_motor8_guessB.npz")

    if not os.path.exists(mu):
        mu   = mu.replace("Subj", "subj")
        beta = beta.replace("Subj", "subj")

    if band == "8-13Hz":
        return mu, mu    # MU seule
    elif band == "13-30Hz":
        return beta, beta  # BETA seule
    else:
        return mu, beta    # Large bande


def build_windows(run):
    """
    Construit les fenêtres d'entrée pour DeepConvNet depuis un run.
    Utilise les clés X_nomove1, X_nomove2 (IDLE) et X_move1, X_move4 (MOVE).
    Les labels sont dans run['y'] : 1=RIGHT, 2=LEFT, 3=WALK.
    Retourne X (n_trials, n_channels, n_samples) et y (labels 0-3).
    """
    X_list, y_list = [], []

    # IDLE : fenêtres de repos (label 0)
    for key in ["X_nomove1", "X_nomove2"]:
        if key in run:
            arr = np.array(run[key], dtype=np.float32)
            if arr.ndim == 3:
                for trial in arr:
                    X_list.append(trial)
                    y_list.append(0)

    # MOVE : fenêtres d'imagerie motrice (labels 1, 2, 3)
    # X_move1..X_move6 correspondent aux trials de la session
    # Les labels réels sont dans run['y']
    y_labels = np.array(run['y']).ravel() if 'y' in run else None
    move_keys = [k for k in run.keys() if k.startswith('X_move')]
    move_keys_sorted = sorted(move_keys, key=lambda k: int(k.replace('X_move','')))

    for i, key in enumerate(move_keys_sorted):
        arr = np.array(run[key], dtype=np.float32)
        if arr.ndim == 3:
            for j, trial in enumerate(arr):
                # Récupérer le label depuis y si disponible
                if y_labels is not None and i < len(y_labels):
                    label = int(y_labels[i])
                else:
                    label = 1  # fallback
                X_list.append(trial)
                y_list.append(label)

    if not X_list:
        return None, None

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y


# ============================================================
# ENTRAÎNEMENT
# ============================================================

def train_deepconvnet(X_tr, y_tr, X_val, y_val, params):
    """
    Entraîne DeepConvNet avec early stopping.

    Args:
        X_tr, y_tr  : données d'entraînement
        X_val, y_val: données de validation pour early stopping
        params      : dict avec dropout et lr

    Returns:
        model entraîné, epoch de early stopping
    """
    n_channels = X_tr.shape[1]
    n_classes  = len(np.unique(y_tr))

    model = DeepConvNet(n_channels=n_channels, n_classes=n_classes,
                        dropout=params["dropout"]).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=params["lr"])
    loss_fn = nn.CrossEntropyLoss()

    # Tenseurs PyTorch
    X_t = torch.tensor(X_tr).to(device)
    y_t = torch.tensor(y_tr).to(device)
    X_v = torch.tensor(X_val).to(device)
    y_v = torch.tensor(y_val).to(device)

    dl = DataLoader(TensorDataset(X_t, y_t),
                    batch_size=BATCH_SIZE, shuffle=True)

    best_val_loss = float('inf')
    patience_cnt  = 0
    best_state    = None
    early_epoch   = FAST_EPOCHS

    for epoch in range(FAST_EPOCHS):
        # Entraînement
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_v), y_v).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt  = 0
        else:
            patience_cnt += 1
            if patience_cnt >= FAST_PATIENCE:
                early_epoch = epoch + 1
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, early_epoch


# ============================================================
# ÉVALUATION LORO PARTIEL
# ============================================================

def evaluate_config(subj, params):
    """
    Évalue une configuration sur un sujet avec LORO partiel (3 folds).
    Retourne la balanced accuracy moyenne sur les 3 folds.
    """
    mu_path, beta_path = get_npz_paths(subj, params["band"])
    if not os.path.exists(mu_path):
        return None

    runs_mu   = load_npz(mu_path)
    runs_beta = load_npz(beta_path)
    n_runs    = len(runs_mu)

    valid_folds = [i for i in LORO_TEST_FOLDS if i < n_runs]
    if not valid_folds:
        return None

    fold_bals = []

    for test_idx in valid_folds:
        train_idx = [i for i in range(n_runs) if i != test_idx]

        # Données d'entraînement
        X_list, y_list = [], []
        for i in train_idx:
            run = {**runs_mu[i], **{f"beta_{k}": v
                   for k, v in runs_beta[i].items()}}
            X_i, y_i = build_windows(runs_mu[i])
            if X_i is not None:
                X_list.append(X_i)
                y_list.append(y_i)

        if not X_list:
            continue

        X_tr = np.concatenate(X_list)
        y_tr = np.concatenate(y_list)

        # Split train/val pour early stopping (80/20)
        n_val   = max(1, int(0.2 * len(X_tr)))
        X_val   = X_tr[-n_val:]
        y_val   = y_tr[-n_val:]
        X_tr    = X_tr[:-n_val]
        y_tr    = y_tr[:-n_val]

        # Données de test
        X_te, y_te = build_windows(runs_mu[test_idx])

        if X_te is None or len(np.unique(y_te)) < 2:
            continue

        # Entraînement
        model, early_ep = train_deepconvnet(X_tr, y_tr, X_val, y_val, params)

        # Évaluation
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X_te).to(device))
            preds  = logits.argmax(1).cpu().numpy()

        bal = balanced_accuracy_score(y_te, preds)
        fold_bals.append(bal)
        print(f"      fold{test_idx+1}: bal_acc={bal*100:.1f}% (early_stop=ep{early_ep})")

    return float(np.mean(fold_bals)) if fold_bals else None


# ============================================================
# MAIN
# ============================================================

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"\n{'='*65}")
    print(f"  RECHERCHE HYPERPARAM — DeepConvNet — LORO PARTIEL")
    print(f"  {len(CONFIGS)} configs × {len(SUBJECTS)} sujets × {len(LORO_TEST_FOLDS)} folds")
    print(f"  epochs={FAST_EPOCHS}, patience={FAST_PATIENCE}")
    print(f"{'='*65}")

    all_rows = []

    for config_name, params in CONFIGS:
        print(f"\n[{config_name}] dropout={params['dropout']} "
              f"lr={params['lr']} band={params['band']}")

        bals = []
        for subj in SUBJECTS:
            print(f"  {subj}:")
            t0 = time.time()
            try:
                bal = evaluate_config(subj, params)
            except Exception as e:
                print(f"    ERREUR: {e}")
                continue
            if bal is None:
                print(f"    données insuffisantes")
                continue
            dt = time.time() - t0
            print(f"  → {subj} bal_acc_moy={bal*100:.1f}% ({dt:.0f}s)")
            bals.append(bal)
            all_rows.append({
                "config"   : config_name,
                "sujet"    : subj,
                "dropout"  : params["dropout"],
                "lr"       : params["lr"],
                "band"     : params["band"],
                "bal_acc"  : round(bal * 100, 1),
            })

        if bals:
            print(f"  → MOYENNE groupe: bal_acc={np.mean(bals)*100:.1f}%")

    # Sauvegarde CSV
    csv_path = os.path.join(OUTPUT_DIR, "results_loro.csv")
    if all_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)

    # Résumé trié
    print(f"\n{'='*65}")
    print("  RÉSUMÉ — Balanced Accuracy moyenne par configuration")
    print(f"{'='*65}")
    summary = {}
    for row in all_rows:
        summary.setdefault(row["config"], []).append(row["bal_acc"])
    ranked = sorted(summary.items(),
                    key=lambda kv: np.mean(kv[1]), reverse=True)
    for name, vals in ranked:
        print(f"  {name:<20} bal_acc={np.mean(vals):.1f}%")

    best = ranked[0]
    print(f"\n  MEILLEURE CONFIG : {best[0]}")
    print(f"  Résultats : {csv_path}")


if __name__ == "__main__":
    main()