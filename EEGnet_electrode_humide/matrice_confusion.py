#!/usr/bin/env python3
"""
EEGNet par sujet — Config Jour13 uniquement, avec matrice de confusion
========================================================================
Version modifiée pour ne lancer QUE la config Jour13 et générer
la matrice de confusion agrégée sur tous les sujets.

Usage :
    python run_eegnet_cm_jour13.py

Résultats dans :
    ./resultats_eegnet/eegnet_jour13_cm.csv
    ./resultats_eegnet/cm_eegnet_jour13.png
"""

import os, sys, csv, time
from collections import defaultdict
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

BASE_DIR   = "/Users/emma/Desktop/Code/code_pipeline"
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "resultats_eegnet")
os.makedirs(OUTPUT_DIR, exist_ok=True)
sys.path.insert(0, MODELS_DIR)

from eegnet_4class import (
    set_seed, load_runs, zscore_runs_independent,
    build_gate_idle_vs_move, build_expert_rl,
    train_gate, train_expert,
    build_test_eval_4class,
    pipeline_predict_4class,
    find_best_thr_on_train,
    confusion_matrix, balanced_accuracy,
)

SEED     = 42
EPOCHS   = 150
PATIENCE = 30
BATCH    = 64
LR       = 5e-4
WD       = 1e-4
AUG      = 9

IDLE_W  = ["nomove1", "nomove2"]
MOVE_G  = ["move1", "move4"]
MOVE_E  = ["move1"]
MODE    = "time"

LABELS4 = ["RIGHT", "LEFT", "WALK", "IDLE"]  # même ordre que le pipeline

SUBJECTS = [
    "Subj04", "Subj05", "subj06", "Subj07", "Subj08",
    "Subj09", "Subj10", "Subj11", "Subj12",
    "Subj15", "Subj17", "Subj18", "Subj19"
]


def get_config_jour13(subj):
    """Ne retourne que le fichier Jour13 pour ce sujet (ou None)."""
    d13 = os.path.join(BASE_DIR, "npz_all_subjects_day13")
    p = os.path.join(d13, f"{subj}_Jour13_band1330_merged.npz")
    return p if os.path.exists(p) else None


def run_config_split(beta_path, device):
    """
    Split simple : tous les runs sauf le dernier → train,
    dernier run → test. Retourne aussi y_true/y_pred pour la matrice.
    """
    runs = zscore_runs_independent(load_runs(beta_path))
    n = len(runs)
    if n < 2:
        return None

    test_idx   = n - 1
    train_runs = runs[:test_idx]

    Xg_list, yg_list = [], []
    Xe_list, ye_list = [], []
    for r in train_runs:
        Xg_i, yg_i = build_gate_idle_vs_move(r, IDLE_W, MOVE_G, MODE)
        Xe_i, ye_i = build_expert_rl(r, r, MOVE_E, MODE)
        Xg_list.append(Xg_i); yg_list.append(yg_i)
        Xe_list.append(Xe_i); ye_list.append(ye_i)

    Xg_tr = np.concatenate(Xg_list)
    yg_tr = np.concatenate(yg_list)
    Xe_tr = np.concatenate(Xe_list)
    ye_tr = np.concatenate(ye_list)

    gate, mu_g, sd_g = train_gate(
        Xg_tr, yg_tr, device, n_classes=2,
        val_split=0.2, seed_split=SEED,
        do_zscore=True, kernel_len=128,
        dropout=0.25, lr=LR, wd=WD,
        batch_size=BATCH, epochs=EPOCHS, patience=PATIENCE,
        F1=8, D=2, F2=16)

    expert, mu_e, sd_e = train_expert(
        Xe_tr, ye_tr, device, n_classes=3,
        val_split=0.2, seed_split=SEED,
        do_zscore=True, kernel_len=192,
        dropout=0.25, lr=LR, wd=WD,
        batch_size=BATCH, epochs=EPOCHS, patience=PATIENCE,
        F1=8, D=2, F2=16, aug_factor=AUG)

    test_run = runs[test_idx]
    Xg_te, Xe_te, y_true, idle_mask, move_mask = build_test_eval_4class(
        test_run, test_run,
        IDLE_W, MOVE_G, MOVE_E, MODE, MODE)

    if test_idx >= 2:
        val_run = runs[test_idx - 1]
        Xg_v, Xe_v, y_v, im_v, mm_v = build_test_eval_4class(
            val_run, val_run, IDLE_W, MOVE_G, MOVE_E, MODE, MODE)
        thr = find_best_thr_on_train(
            gate, expert, Xg_v, Xe_v, y_v, im_v, mm_v,
            device, True, True, mu_g, sd_g, mu_e, sd_e)
    else:
        thr = 0.30

    y_pred = pipeline_predict_4class(
        gate, mu_g, sd_g, expert, mu_e, sd_e,
        Xg_te, Xe_te, idle_mask, move_mask,
        device, thr, True, True, smooth_k=5)

    cm  = confusion_matrix(y_true, y_pred, 4)
    acc = float((y_true == y_pred).mean())
    bal = balanced_accuracy(cm)
    return round(acc * 100, 1), round(bal * 100, 1), y_true, y_pred


def main():
    set_seed(SEED)
    device = torch.device("cpu")

    print(f"\n{'='*60}")
    print(f"  EEGNet par sujet — Jour13 uniquement — matrice de confusion")
    print(f"{'='*60}")

    all_rows = []
    y_true_all_subjects = []
    y_pred_all_subjects = []

    for subj in SUBJECTS:
        beta_path = get_config_jour13(subj)
        if not beta_path:
            print(f"\n  {subj}: fichier Jour13 introuvable — ignoré")
            continue

        print(f"\n  {subj}: Jour13... ", end="", flush=True)
        t0 = time.time()
        try:
            result = run_config_split(beta_path, device)
        except Exception as e:
            print(f"ERREUR: {e}")
            continue
        if result is None:
            print("données insuffisantes")
            continue

        acc, bal, y_true, y_pred = result
        dt = time.time() - t0
        print(f"acc={acc}%  bal={bal}%  ({dt:.0f}s)")

        all_rows.append({"sujet": subj, "config": "Jour13", "acc": acc, "bal_acc": bal})
        y_true_all_subjects.append(y_true)
        y_pred_all_subjects.append(y_pred)

    # ── Sauvegarde CSV ──
    csv_path = os.path.join(OUTPUT_DIR, "eegnet_jour13_cm.csv")
    if all_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["sujet", "config", "acc", "bal_acc"])
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\n  Résultats CSV : {csv_path}")

    # ── Matrice de confusion agrégée ──
    if y_true_all_subjects:
        y_true_agg = np.concatenate(y_true_all_subjects)
        y_pred_agg = np.concatenate(y_pred_all_subjects)
        cm = confusion_matrix(y_true_agg, y_pred_agg, 4)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS4)
        fig, ax = plt.subplots(figsize=(5, 5))
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format='d')
        ax.set_title("EEGNet — Config Jours 1+3 (tous sujets)", fontsize=12, fontweight="bold")
        plt.tight_layout()

        cm_path = os.path.join(OUTPUT_DIR, "cm_eegnet_jour13.png")
        plt.savefig(cm_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Matrice de confusion : {cm_path}")
    else:
        print("\n  Aucun résultat — matrice non générée")

    # ── Résumé ──
    print(f"\n{'='*60}")
    print("  RÉSUMÉ")
    print(f"{'='*60}")
    if all_rows:
        accs = [r["acc"] for r in all_rows]
        print(f"  Jour13   acc_moy={np.mean(accs):.1f}%  (n={len(accs)} sujets)")


if __name__ == "__main__":
    main()