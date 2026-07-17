#!/usr/bin/env python3
"""
hyperparam_search_fast.py — Recherche d'hyperparamètres EEGNet (LORO partiel)
==============================================================================
Développé dans le cadre du projet BCI-EEG ÉTS (2026)
Auteure : Emma

OBJECTIF :
    Ce script identifie la configuration optimale des hyperparamètres
    d'EEGNet pour les données à électrodes humides, en testant une
    variable à la fois par rapport à une configuration de base.
    Les résultats ont permis de sélectionner la configuration retenue
    dans le Chapitre 4 (section 4.2.2) du mémoire.

PROTOCOLE D'ÉVALUATION :
    Pour limiter le temps de calcul tout en restant rigoureux, on utilise
    un LORO partiel : 3 folds seulement (runs 1, 5 et 10 comme runs de
    test à tour de rôle) sur 3 sujets représentatifs (Subj04, Subj05,
    Subj07), avec 60 epochs maximum et une patience de 15.
    La configuration retenue est celle donnant la meilleure balanced
    accuracy moyenne sur les 3 sujets × 3 folds = 9 entraînements.

CONFIGURATIONS TESTÉES (une variable à la fois) :
    - Configuration de base : F1=8, dropout=0.25, lr=5e-4,
      kernel_gate=128, kernel_expert=192, bande=8-30Hz
    - F1 : 4, 8 (base), 16
    - Dropout : 0.25 (base), 0.5
    - Learning rate : 1e-3, 5e-4 (base), 1e-4
    - Kernel gate : 64, 128 (base)
    - Kernel expert : 128, 192 (base)
    - Bande de fréquence : 8-30Hz (base), 8-13Hz (mu), 13-30Hz (bêta)

RÉSULTAT RETENU :
    La bande bêta seule (13-30Hz) s'est révélée la meilleure configuration
    avec une balanced accuracy moyenne de 55.0% (vs 52.5% pour la base).
    → Voir résultats complets dans hyperparam_results_fast/results_loro.csv

USAGE :
    python hyperparam_search_fast.py

DÉPENDANCES :
    pip install numpy torch scikit-learn
    (eegnet_4class.py doit être dans le dossier models/)
"""
import os, sys, csv, time
import numpy as np
import torch

BASE_DIR   = "/Users/emma/Desktop/Code/code_pipeline"
NPZ_DAY1   = os.path.join(BASE_DIR, "npz_all_subjects_day1")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "hyperparam_results_fast")
os.makedirs(OUTPUT_DIR, exist_ok=True)
sys.path.insert(0, MODELS_DIR)

from eegnet_4class import (
    set_seed, load_runs,
    build_gate_idle_vs_move, build_expert_rl,
    train_gate, train_expert,
    build_test_eval_4class,
    pipeline_predict_4class,
    confusion_matrix, balanced_accuracy,
)

# ── Sujets ───────────────────────────────────────────────────────────────────
SUBJECTS = ["Subj04", "Subj05", "Subj07"]

# ── Folds LORO partiels : indices des runs utilisés comme test ───────────────
LORO_TEST_FOLDS = [0, 4, 9]   # runs 1, 5, 10 (index 0-based)

# ── Configurations à tester ──────────────────────────────────────────────────
BASE = {
    "F1": 8, "dropout": 0.25, "lr": 5e-4,
    "kernel_gate": 128, "kernel_expert": 192, "band": "8-30Hz",
}

CONFIGS = [
    ("base",              {**BASE}),
    ("F1-4",              {**BASE, "F1": 4}),
    ("F1-16",             {**BASE, "F1": 16}),
    ("dropout-05",        {**BASE, "dropout": 0.5}),
    ("lr-1e3",            {**BASE, "lr": 1e-3}),
    ("lr-1e4",            {**BASE, "lr": 1e-4}),
    ("kernel_gate-64",    {**BASE, "kernel_gate": 64}),
    ("kernel_expert-128", {**BASE, "kernel_expert": 128}),
    ("band-mu",           {**BASE, "band": "8-13Hz"}),
    ("band-beta",         {**BASE, "band": "13-30Hz"}),
]

# ── Hyperparamètres fixes ─────────────────────────────────────────────────────
FAST_EPOCHS         = 60
FAST_PATIENCE       = 15
VAL_SPLIT           = 0.2
BATCH_SIZE          = 64
WD                  = 1e-4
SEED                = 42
IDLE_WINDOWS        = ["nomove1", "nomove2"]
MOVE_WINDOWS_GATE   = ["move1", "move4"]
MOVE_WINDOWS_EXPERT = ["move1"]
STACK_MODE          = "time"
EXPERT_AUG_FACTOR   = 5


def get_npz_paths(subj):
    mu   = os.path.join(NPZ_DAY1, f"{subj}_Jour1_band0812_motor8_guessB.npz")
    beta = os.path.join(NPZ_DAY1, f"{subj}_Jour1_band1330_motor8_guessB.npz")
    if not os.path.exists(mu):
        mu   = mu.replace("Subj", "subj")
        beta = beta.replace("Subj", "subj")
    return mu, beta


def load_subject_runs(subj, band):
    mu_path, beta_path = get_npz_paths(subj)
    runs_mu = load_runs(mu_path)
    runs_b  = load_runs(beta_path)
    if band == "8-13Hz":
        runs_b = runs_mu
    elif band == "13-30Hz":
        runs_mu = runs_b
    return runs_mu, runs_b


def evaluate_config_loro_partial(subj, params, device):
    """
    LORO partiel : entraîne et teste sur 3 folds sélectionnés.
    Retourne la balanced accuracy moyenne sur les 3 folds.
    """
    runs_mu, runs_b = load_subject_runs(subj, params["band"])
    n_runs = len(runs_mu)

    # Filtrer les folds valides (index dans les bornes)
    valid_folds = [i for i in LORO_TEST_FOLDS if i < n_runs]
    if len(valid_folds) == 0:
        return None

    fold_bals = []

    for test_idx in valid_folds:
        train_idx = [i for i in range(n_runs) if i != test_idx]
        if len(train_idx) == 0:
            continue

        # Construire données d'entraînement
        Xg_list, yg_list = [], []
        Xe_list, ye_list = [], []
        for i in train_idx:
            Xg_i, yg_i = build_gate_idle_vs_move(
                runs_mu[i], IDLE_WINDOWS, MOVE_WINDOWS_GATE, STACK_MODE)
            Xe_i, ye_i = build_expert_rl(
                runs_mu[i], runs_b[i], MOVE_WINDOWS_EXPERT, STACK_MODE)
            Xg_list.append(Xg_i); yg_list.append(yg_i)
            Xe_list.append(Xe_i); ye_list.append(ye_i)

        Xg_tr = np.concatenate(Xg_list)
        yg_tr = np.concatenate(yg_list)
        Xe_tr = np.concatenate(Xe_list)
        ye_tr = np.concatenate(ye_list)

        # Entraîner gate
        gate, mu_g, sd_g = train_gate(
            Xg_tr, yg_tr, device, n_classes=2,
            val_split=VAL_SPLIT, seed_split=SEED,
            do_zscore=True, kernel_len=params["kernel_gate"],
            dropout=params["dropout"], lr=params["lr"], wd=WD,
            batch_size=BATCH_SIZE, epochs=FAST_EPOCHS, patience=FAST_PATIENCE,
            F1=params["F1"], D=2, F2=16)

        # Entraîner expert
        expert, mu_e, sd_e = train_expert(
            Xe_tr, ye_tr, device, n_classes=3,
            val_split=VAL_SPLIT, seed_split=SEED,
            do_zscore=True, kernel_len=params["kernel_expert"],
            dropout=params["dropout"], lr=params["lr"], wd=WD,
            batch_size=BATCH_SIZE, epochs=FAST_EPOCHS, patience=FAST_PATIENCE,
            F1=params["F1"], D=2, F2=16, aug_factor=EXPERT_AUG_FACTOR)

        # Évaluer sur le run test
        Xg_te, Xe_te, y_true, idle_mask, move_mask = build_test_eval_4class(
            runs_mu[test_idx], runs_b[test_idx],
            IDLE_WINDOWS, MOVE_WINDOWS_GATE, MOVE_WINDOWS_EXPERT,
            STACK_MODE, STACK_MODE)

        y_pred = pipeline_predict_4class(
            gate, mu_g, sd_g, expert, mu_e, sd_e,
            Xg_te, Xe_te, idle_mask, move_mask,
            device, 0.30, True, True, smooth_k=5)

        cm  = confusion_matrix(y_true, y_pred, 4)
        bal = balanced_accuracy(cm)
        acc = float((y_true == y_pred).mean())
        fold_bals.append(bal)

        print(f"      fold{test_idx+1}: acc={acc*100:.1f}% bal={bal*100:.1f}%")

    return float(np.mean(fold_bals)) if fold_bals else None


def main():
    set_seed(SEED)
    device = torch.device("cpu")
    t0 = time.time()

    print(f"\n{'='*65}")
    print(f"  RECHERCHE HYPERPARAM — LORO PARTIEL (folds 1,5,10)")
    print(f"  {len(CONFIGS)} configs × {len(SUBJECTS)} sujets × {len(LORO_TEST_FOLDS)} folds")
    print(f"  = {len(CONFIGS)*len(SUBJECTS)*len(LORO_TEST_FOLDS)} entraînements")
    print(f"  epochs={FAST_EPOCHS}, patience={FAST_PATIENCE}")
    print(f"{'='*65}")

    all_rows = []

    for config_name, params in CONFIGS:
        print(f"\n[{config_name}] F1={params['F1']} do={params['dropout']} "
              f"lr={params['lr']} kg={params['kernel_gate']} "
              f"ke={params['kernel_expert']} band={params['band']}")

        bals = []
        for subj in SUBJECTS:
            print(f"  {subj}:")
            t1 = time.time()
            try:
                bal = evaluate_config_loro_partial(subj, params, device)
            except Exception as e:
                print(f"    ERREUR: {e}")
                continue
            if bal is None:
                continue
            dt = time.time() - t1
            print(f"  → {subj} bal_acc_moy={bal*100:.1f}% ({dt:.0f}s)")
            bals.append(bal)
            all_rows.append({
                "config": config_name, "sujet": subj,
                **{k: v for k, v in params.items()},
                "bal_acc_mean_folds": bal,
            })

        if bals:
            print(f"  → MOYENNE groupe: bal_acc={np.mean(bals)*100:.1f}%")

    total = time.time() - t0
    print(f"\nTemps total : {total/60:.1f} minutes")

    # Sauvegarde
    csv_path = os.path.join(OUTPUT_DIR, "results_loro.csv")
    if all_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)

    # Résumé trié
    print(f"\n{'='*65}")
    print("  RÉSUMÉ — Balanced Accuracy moyenne (LORO partiel)")
    print(f"{'='*65}")
    summary = {}
    for row in all_rows:
        summary.setdefault(row["config"], []).append(row["bal_acc_mean_folds"])
    ranked = sorted(summary.items(),
                    key=lambda kv: np.mean(kv[1]), reverse=True)
    for name, vals in ranked:
        print(f"  {name:<22} bal_acc={np.mean(vals)*100:.1f}% "
              f"(±{np.std(vals)*100:.1f}%)")

    best = ranked[0]
    print(f"\n  MEILLEURE CONFIG : {best[0]}")
    print(f"  Résultats : {csv_path}")


if __name__ == "__main__":
    main()
