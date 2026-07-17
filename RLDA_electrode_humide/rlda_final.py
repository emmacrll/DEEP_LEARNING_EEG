#!/usr/bin/env python3
"""
rlda_final.py — Pipeline de classification RLDA sur données EEG humides
========================================================================
Développé dans le cadre du projet BCI-EEG ÉTS (2026)
Auteure : Emma

OBJECTIF :
    Ce script reproduit le pipeline de classification RLDA d'Alchalabi
    et al. (2021) et l'applique aux données EEG à électrodes humides
    pour tous les sujets et toutes les configurations de sessions.
    Il a permis de produire les résultats du Tableau 5.1.1 du mémoire.

CHOIX D'IMPLÉMENTATION :
    - Features PSDAG (645 dimensions) recalculées depuis le signal brut
    - Filtre IIR A1013 (bande 10-13 Hz, upper mu) hérité du pipeline original
    - Fenêtre temporelle move1 (échantillons 1050-1306, ~1 seconde d'imagerie)
    - Classifieur LDA avec shrinkage fixé à 0.4 (Ledoit-Wolf)
    - Validation croisée stratifiée à 10 plis
    - 4 classes : IDLE (0), RIGHT (1), LEFT (2), WALK (3)
    - 7 configurations de sessions testées : J1, J2, J3, J1+2, J1+3, J2+3, J1+2+3

FORMAT DES DONNÉES :
    Fichiers MATLAB (.mat) organisés par sujet et par session :
        base/SubjXX/Jour 1/*BCI_Run_Training.mat
        base/SubjXX/Jour 2/*BCI_Run_Test.mat
        base/SubjXX/jour 3/*BCI_Run_Test.mat

    ⚠️  Les noms de dossiers varient selon les sujets (casse, espaces).
    Le script détecte automatiquement les variantes communes.

USAGE :
    # Tous les sujets d'un coup (résultats sauvegardés en CSV)
    python rlda_final.py --all \\
        --base "/chemin/vers/data/etude 2" \\
        --iir  "/chemin/vers/BCI_IIR_Filters.mat"

    # Un seul sujet
    python rlda_final.py \\
        --subj Subj05 \\
        --j1 "/chemin/vers/Subj05/Jour 1" \\
        --j2 "/chemin/vers/Subj05/Jour 2" \\
        --j3 "/chemin/vers/Subj05/jour 3" \\
        --iir "/chemin/vers/BCI_IIR_Filters.mat"

DÉPENDANCES :
    pip install numpy scipy scikit-learn
"""

import argparse
import numpy as np
import glob
import csv
import os
from pathlib import Path
from scipy.io import loadmat
from scipy.signal import sosfiltfilt, welch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold

# ============================================================
# CONSTANTES GLOBALES
# ============================================================

CHAN      = [1, 7, 8, 13, 14, 4, 5, 6]
N_FEAT    = 645
WIN_SIZE  = 256
MOVE_START = 1050

# Sujets disponibles pour le mode --all
ALL_SUBJECTS = [
    "Subj04", "Subj05", "subj06", "Subj07", "Subj08",
    "Subj09", "Subj10", "Subj11", "Subj12",
    "Subj15", "Subj17", "Subj18", "Subj19"
]

# Variantes de noms de dossiers jour
JOUR_VARIANTS = {
    "j1": ["Jour 1", "jour 1", "Jour1", "jour1"],
    "j2": ["Jour 2", "jour 2", "Jour2", "jour2"],
    "j3": ["Jour 3", "jour 3", "Jour3", "jour3"],
}


# ============================================================
# CHARGEMENT DU FILTRE
# ============================================================

def load_iir(iir_path):
    iir = loadmat(str(iir_path))
    A = np.asarray(iir['A1013'], dtype=np.float64)
    return np.column_stack([A[:, :3], A[:, 3:]]).copy()


# ============================================================
# CALCUL DES FEATURES PSDAG
# ============================================================

def compute_psdag(win_8ch, sos):
    if win_8ch.shape[1] < WIN_SIZE:
        return None
    filt = sosfiltfilt(sos, win_8ch.astype(np.float64), axis=1)
    _, psd = welch(filt, fs=256, nperseg=WIN_SIZE, axis=1)
    eps = 1e-10
    r1 = (psd[1] - psd[2]) / (psd[1] + psd[2] + eps)
    r2 = psd[0]
    r3 = psd[5]
    r4 = (psd[3] - psd[4]) / (psd[3] + psd[4] + eps)
    r5 = (psd[6] - psd[7]) / (psd[6] + psd[7] + eps)
    feat = np.concatenate([r1, r2, r3, r4, r5])
    if feat.shape[0] == N_FEAT:
        return feat
    return None


def unwrap_cell(cell):
    while hasattr(cell, 'ravel') and cell.dtype == object:
        cell = cell.ravel()[0]
    return cell


# ============================================================
# CHARGEMENT D'UN JOUR
# ============================================================

def load_day(folder, sos):
    folder = str(folder)
    files_te = sorted(glob.glob(folder + "/*BCI_Run_Test.mat"))
    files_tr = sorted(glob.glob(folder + "/*BCI_Run_Training.mat"))
    files = files_te if files_te else files_tr
    is_test = bool(files_te)
    X_all, y_all = [], []

    for f in files:
        mat = loadmat(f, squeeze_me=False, struct_as_record=False)
        TR   = mat["Trials_Matrix_Cue"][:, 0].astype(int)
        data = mat["Data"].ravel()
        n    = min(len(TR), len(data))

        for i in range(n):
            cell  = unwrap_cell(data[i])
            label = int(TR[i])
            if label not in [1, 2, 3]:
                continue
            try:
                if is_test:
                    bl = np.array(cell.baseline,  dtype=np.float64)
                    an = np.array(cell.animation, dtype=np.float64)
                    if bl.shape[0] > bl.shape[1]: bl = bl.T
                    if an.shape[0] > an.shape[1]: an = an.T
                    raw = np.zeros((20, 2000))
                    raw[:, :min(bl.shape[1], 650)] = bl[:, :min(bl.shape[1], 650)]
                    raw[:, 1050:1050 + min(an.shape[1], 950)] = an[:, :min(an.shape[1], 950)]
                else:
                    raw = np.array(cell.data, dtype=np.float64)
                    if raw.shape[0] > raw.shape[1]: raw = raw.T
                    if raw.shape[1] < 2000: continue

                raw_ch = raw[CHAN, :2000]

                feat_m = compute_psdag(raw_ch[:, MOVE_START:MOVE_START + WIN_SIZE], sos)
                if feat_m is not None:
                    X_all.append(feat_m.astype(np.float32))
                    y_all.append(label)

                nm1 = compute_psdag(raw_ch[:, 250:506], sos)
                nm2 = compute_psdag(raw_ch[:, 400:656], sos)
                if nm1 is not None:
                    X_all.append(nm1.astype(np.float32)); y_all.append(0)
                if nm2 is not None:
                    X_all.append(nm2.astype(np.float32)); y_all.append(0)
            except:
                pass

    if not X_all:
        return np.zeros((0, N_FEAT)), np.zeros(0, dtype=int)
    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=int)
    valid = np.isfinite(X).all(axis=1)
    return X[valid], y[valid]


# ============================================================
# ÉVALUATION RLDA — VALIDATION CROISÉE 10 PLIS
# ============================================================

def eval_cv10(X, y):
    """
    Évalue le classifieur RLDA par validation croisée à 10 plis.

    Retourne :
        acs   : accuracy BCI_STEPS (classes 1,2 vs 0)
        acw   : accuracy BCI_WALK  (classe 3 vs 0)
        mean  : moyenne des deux
        std_s : écart-type BCI_STEPS
        std_w : écart-type BCI_WALK
    """
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.4)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # BCI_STEPS : distinguer pas gauche/droit du repos
    mask_s = np.isin(y, [0, 1, 2])
    Xs, ys = X[mask_s], y[mask_s]
    scores_s = []
    if len(np.unique(ys)) >= 2 and len(ys) >= 20:
        for tr, te in skf.split(Xs, ys):
            clf.fit(Xs[tr], ys[tr])
            scores_s.append(clf.score(Xs[te], ys[te]))
    acs   = float(np.mean(scores_s)) if scores_s else 0.0
    std_s = float(np.std(scores_s))  if scores_s else 0.0

    # BCI_WALK : distinguer marche du repos
    mask_w = np.isin(y, [0, 3])
    Xw, yw = X[mask_w], y[mask_w]
    scores_w = []
    if len(np.unique(yw)) >= 2 and len(yw) >= 20:
        for tr, te in skf.split(Xw, yw):
            clf.fit(Xw[tr], yw[tr])
            scores_w.append(clf.score(Xw[te], yw[te]))
    acw   = float(np.mean(scores_w)) if scores_w else 0.0
    std_w = float(np.std(scores_w))  if scores_w else 0.0

    mean = (acs + acw) / 2.0
    return round(acs, 4), round(acw, 4), round(mean, 4), round(std_s, 4), round(std_w, 4)


# ============================================================
# PIPELINE PAR SUJET
# ============================================================

def find_jour(subj_path, variants):
    """Trouve le dossier jour parmi les variantes possibles."""
    for v in variants:
        p = os.path.join(subj_path, v)
        if os.path.isdir(p):
            return p
    return None


def run_subject(subj_id, day_folders, sos, csv_writer=None):
    print("\n" + "=" * 60)
    print("SUJET :", subj_id)
    print("=" * 60)

    days = []
    for d, folder in enumerate(day_folders):
        label = "Jour " + str(d + 1)
        if folder is None or not Path(folder).exists():
            print(label, ": MANQUANT")
            days.append((np.zeros((0, N_FEAT)), np.zeros(0)))
            continue
        X, y = load_day(folder, sos)
        if len(X) == 0:
            print(label, ": VIDE")
            days.append((np.zeros((0, N_FEAT)), np.zeros(0)))
            continue
        print(label, ":", len(X), "trials")
        days.append((X, y))

    Xj = [d[0] for d in days]
    yj = [d[1] for d in days]

    def merge(idxs):
        Xs = [Xj[i] for i in idxs if len(Xj[i]) > 0]
        ys = [yj[i] for i in idxs if len(yj[i]) > 0]
        if not Xs:
            return np.zeros((0, N_FEAT)), np.zeros(0)
        return np.concatenate(Xs), np.concatenate(ys)

    configs = {
        "1":     merge([0]),
        "2":     merge([1]),
        "1+2":   merge([0, 1]),
        "3":     merge([2]),
        "1+3":   merge([0, 2]),
        "2+3":   merge([1, 2]),
        "1+2+3": merge([0, 1, 2]),
    }

    for lbl, (X, y) in configs.items():
        if len(X) < 20:
            continue
        acs, acw, mean, std_s, std_w = eval_cv10(X, y)
        print(lbl, "STEPS:", acs, "WALK:", acw, "MEAN:", mean)

        # Sauvegarde CSV si writer fourni
        if csv_writer is not None:
            csv_writer.writerow({
                "sujet":  subj_id,
                "config": lbl,
                "steps":  round(acs * 100, 1),
                "walk":   round(acw * 100, 1),
                "mean":   round(mean * 100, 1),
            })

    return configs


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subj", type=str)
    ap.add_argument("--j1",   type=str)
    ap.add_argument("--j2",   type=str)
    ap.add_argument("--j3",   type=str)
    ap.add_argument("--iir",  type=str)
    ap.add_argument("--all",  action="store_true")
    ap.add_argument("--base", type=str,
                    default="/Users/emma/Desktop/copy/data/etude 2")
    ap.add_argument("--out",  type=str,
                    default="/Users/emma/Desktop/Code/code_RDLA/resultats_rlda/rlda_par_sujet.csv")
    args = ap.parse_args()

    sos = load_iir(args.iir)

    if args.all:
        # Mode ALL : boucle sur tous les sujets
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", newline="") as f:
            fieldnames = ["sujet", "config", "steps", "walk", "mean"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for subj in ALL_SUBJECTS:
                subj_path = os.path.join(args.base, subj)
                if not os.path.isdir(subj_path):
                    print(f"\n{subj}: dossier introuvable — ignoré")
                    continue
                j1 = find_jour(subj_path, JOUR_VARIANTS["j1"])
                j2 = find_jour(subj_path, JOUR_VARIANTS["j2"])
                j3 = find_jour(subj_path, JOUR_VARIANTS["j3"])
                run_subject(subj, [j1, j2, j3], sos, csv_writer=writer)

        print(f"\nRésultats sauvegardés : {args.out}")

    else:
        # Mode sujet unique
        folders = [args.j1, args.j2, args.j3]
        run_subject(args.subj, folders, sos)


if __name__ == "__main__":
    main()
