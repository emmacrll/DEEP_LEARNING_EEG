#!/usr/bin/env python3
"""
analyse_quantitative.py — Analyse quantitative ET qualitative complète
des données EEG humides (Alchalabi et al., 2021) pour tous les sujets.
=======================================================================
Développé dans le cadre du projet BCI-EEG ÉTS (2026)
Auteure : Emma

OBJECTIF :
    Ce script analyse la qualité des données EEG et les performances de
    classification RLDA pour tous les sujets et toutes les sessions des
    données à électrodes humides. Il a permis de produire :
    - Le Tableau-A I 2 (Annexe I) : qualité des données par sujet/jour
    - La Figure 32 : synthèse globale des performances RLDA
    - Le fichier analyse_complete_bilal.csv : résultats complets

PRODUIT POUR CHAQUE SUJET :
    - Statistiques par canal (amplitude, std, % trials valides, artefacts)
    - Performances RLDA (BCI_STEPS, BCI_WALK, MEAN) par validation croisée
    - Graphiques : signal brut, PSD, barres STD, progression RLDA par jour
    - CSV global avec tous les résultats pour tous les sujets

FORMAT DES DONNÉES ATTENDU :
    Les données doivent être des fichiers MATLAB (.mat) exportés depuis
    EEGStudio/EEGLab, organisés comme suit :
        base/
        ├── Subj05/
        │   ├── Jour 1/   → fichiers *BCI_Run_Training.mat
        │   ├── Jour 2/   → fichiers *BCI_Run_Test.mat
        │   └── jour 3/   → fichiers *BCI_Run_Test.mat
        ├── Subj07/
        │   └── ...

    IMPORTANT : les noms de dossiers (Jour 1, JOUR 2, Jour2, etc.) varient
    selon les sujets — ils sont spécifiés dans le dictionnaire SUBJECTS
    de la section CONFIGURATION ci-dessous.

DISTINCTION TRAINING vs TEST :
    Les fichiers Training (Jour 1) contiennent le signal EEG continu (cell.data).
    Les fichiers Test (Jours 2-3) contiennent des segments prédécoupés
    (cell.baseline + cell.animation). Le script détecte automatiquement
    le format et adapte le chargement en conséquence.

USAGE :
    # Lancer avec les chemins par défaut (configurés dans CONFIGURATION)
    python analyse_quantitative.py

    # Avec chemins personnalisés
    python analyse_quantitative.py \
        --base "/chemin/vers/data/etude 2/" \
        --out  "/chemin/vers/resultats/"

DÉPENDANCES :
    pip install numpy pandas matplotlib scipy scikit-learn
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import glob
import csv
import os
from pathlib import Path
from scipy.io import loadmat
from scipy.signal import sosfiltfilt, welch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION — Modifier selon votre environnement
# ============================================================

# Dictionnaire des sujets analysés.
# Clé = nom du dossier sujet, Valeur = liste des noms de dossiers de session.
# ⚠️  Les noms de dossiers varient selon les sujets (casse, espaces)
# car les données ont été sauvegardées de façon inconsistante.
# Vérifier avec : ls "/chemin/vers/data/etude 2/SubjXX/"
SUBJECTS = {
    "Subj05": ["Jour 1",  "Jour 2 ", "jour 3 "],
    "subj06": ["Jour 1",  "JOUR 2",  "JOUR 3"],
    "Subj07": ["Jour 1",  "JOUR 2",  "JOUR 3"],
    "Subj09": ["Jour 1 ", "Jour 2 ", "Jour 3 "],
    "Subj10": ["Jour 1",  "Jour 2",  "Jour 3"],
    "Subj11": ["Jour 1",  "Jour2",   "Jour3"],
    "Subj12": ["Jour 1",  "JOUR 2",  "JOUR 3"],
    "Subj17": ["Jour 1",  "Jour2",   "Jour3"],
    "Subj18": ["Jour 1",  "Jour 2",  "Jour 3"],
}

# Indices des canaux EEG dans les fichiers .mat (0-based dans Python)
# Correspondent aux électrodes : Fz, C3, C4, Cz, FC3, FC4, CP3, CP4
# Ces 8 canaux ont été sélectionnés car ils couvrent les régions motrices
# et frontales pertinentes pour l'imagerie motrice de la marche
CHAN       = [1, 7, 8, 13, 14, 4, 5, 6]
CHAN_NAMES = ["Fz", "C3", "C4", "Cz", "FC3", "FC4", "CP3", "CP4"]

# Canaux critiques pour l'imagerie motrice des membres inférieurs
# C3 = hémisphère gauche, C4 = hémisphère droit, Cz = ligne médiane
CRITICAL   = ["C3", "C4", "Cz"]

# Fréquence d'échantillonnage du système Smart BCI (Hz)
FS         = 256

# Seuil de détection des artefacts (µV)
# Les segments dépassant ce seuil sont exclus de l'analyse
AMP_THRESH = 150.0

# ⚠️  MODIFIER CE CHEMIN vers le fichier de filtre IIR sur votre machine
# Ce fichier contient le filtre A1013 (bande 10-13 Hz, upper mu)
# utilisé pour l'extraction des features PSDAG du pipeline RLDA
IIR_PATH   = "/Users/emma/Desktop/copy/Bilal_Scripts copy/Matlab_protocole_EEG_BCI/BCI_IIR_Filters.mat"

# ============================================================
# UTILITAIRES
# ============================================================

def unwrap_cell(cell):
    """
    Déroule les cellules imbriquées des fichiers MATLAB.

    Les fichiers .mat exportés depuis MATLAB contiennent parfois des
    structures de données imbriquées (cell arrays dans cell arrays).
    Cette fonction extrait récursivement la valeur finale jusqu'à
    obtenir un objet numpy exploitable.

    Args:
        cell : objet numpy issu de loadmat()
    Returns:
        L'objet numpy terminal (struct MATLAB déballée)
    """
    while hasattr(cell, 'ravel') and cell.dtype == object:
        cell = cell.ravel()[0]
    return cell


def load_iir():
    """
    Charge le filtre IIR A1013 depuis le fichier MATLAB de Bilal Alchalabi.

    Le filtre A1013 est un filtre IIR passe-bande centré sur la bande
    10-13 Hz (mu supérieure), optimisé empiriquement sur les données
    de l'étude originale (Alchalabi et al., 2021). Il est stocké sous
    forme de coefficients SOS (sections de second ordre) dans le fichier
    BCI_IIR_Filters.mat.

    Les coefficients sont restructurés au format SOS attendu par
    scipy.signal.sosfiltfilt (colonnes b0,b1,b2,a0,a1,a2).

    Returns:
        Matrice SOS (numpy array) ou None si le fichier est introuvable.
        Si None, le calcul RLDA est désactivé pour ce sujet.
    """
    try:
        iir = loadmat(IIR_PATH)
        A   = np.asarray(iir['A1013'], dtype=np.float64)
        return np.column_stack([A[:,:3], A[:,3:]]).copy()
    except:
        return None


def load_day_raw(folder):
    """Charge un jour — retourne signaux bruts + labels."""
    folder   = str(folder)
    files_te = sorted(glob.glob(folder + "/*BCI_Run_Test.mat"))
    files_tr = sorted(glob.glob(folder + "/*BCI_Run_Training.mat"))
    files    = files_te if files_te else files_tr
    is_test  = bool(files_te)

    trials = []
    n_runs = 0
    for f in files:
        n_runs += 1
        try:
            mat  = loadmat(f, squeeze_me=False, struct_as_record=False)
            TR   = mat["Trials_Matrix_Cue"][:,0].astype(int)
            data = mat["Data"].ravel()
            n    = min(len(TR), len(data))
            for i in range(n):
                cell  = unwrap_cell(data[i])
                label = int(TR[i])
                if label not in [0,1,2,3]: continue
                try:
                    if is_test:
                        bl = np.array(cell.baseline, dtype=np.float64)
                        an = np.array(cell.animation, dtype=np.float64)
                        if bl.shape[0]>bl.shape[1]: bl=bl.T
                        if an.shape[0]>an.shape[1]: an=an.T
                        raw = np.zeros((20,2000))
                        raw[:,:min(bl.shape[1],650)]         = bl[:,:min(bl.shape[1],650)]
                        raw[:,1050:1050+min(an.shape[1],950)]= an[:,:min(an.shape[1],950)]
                    else:
                        raw = np.array(cell.data, dtype=np.float64)
                        if raw.shape[0]>raw.shape[1]: raw=raw.T
                        if raw.shape[1]<2000: continue
                    raw_ch = raw[CHAN,:2000]
                    trials.append((raw_ch, label, is_test))
                except: pass
        except: pass
    return trials, n_runs, is_test


# ============================================================
# ANALYSE QUANTITATIVE
# ============================================================

def analyse_quantitative(trials, sos):
    """Stats par canal + performances RLDA."""
    if not trials:
        return None

    # Stats signal par canal
    chan_stats = []
    for ci, cname in enumerate(CHAN_NAMES):
        vals = np.concatenate([t[0][ci] for t in trials if t[1] != 0])
        std_v  = float(np.std(vals))
        mean_v = float(np.mean(vals))
        max_v  = float(np.abs(vals).max())
        n_art  = int((np.abs(vals) > AMP_THRESH).sum())
        pct_v  = float(100*(len(vals)-n_art)/len(vals))
        chan_stats.append({
            'nom': cname, 'mean': mean_v, 'std': std_v,
            'max': max_v, 'n_art': n_art, 'pct_valid': pct_v,
            'is_critical': cname in CRITICAL
        })

    # Comptage labels
    labels = np.array([t[1] for t in trials])
    counts = {l: int((labels==l).sum()) for l in [0,1,2,3]}

    # Performances RLDA
    acs, acw, mean_p = np.nan, np.nan, np.nan
    if sos is not None:
        X_all, y_all = [], []
        for raw_ch, label, _ in trials:
            if label not in [1,2,3]: continue
            try:
                win = raw_ch[:, 1050:1306]
                if win.shape[1] < 256: continue
                filt = sosfiltfilt(sos, win.astype(np.float64), axis=1)
                _, psd = welch(filt, fs=FS, nperseg=256, axis=1)
                eps = 1e-10
                r1=(psd[1]-psd[2])/(psd[1]+psd[2]+eps)
                r2=psd[0]; r3=psd[5]
                r4=(psd[3]-psd[4])/(psd[3]+psd[4]+eps)
                r5=(psd[6]-psd[7])/(psd[6]+psd[7]+eps)
                feat = np.concatenate([r1,r2,r3,r4,r5])
                if feat.shape[0]!=645: continue
                X_all.append(feat.astype(np.float32)); y_all.append(label)
                for ns in [250,400]:
                    win_nm = raw_ch[:,ns:ns+256]
                    if win_nm.shape[1]<256: continue
                    filt_nm = sosfiltfilt(sos, win_nm.astype(np.float64), axis=1)
                    _, psd_nm = welch(filt_nm, fs=FS, nperseg=256, axis=1)
                    r1n=(psd_nm[1]-psd_nm[2])/(psd_nm[1]+psd_nm[2]+eps)
                    r4n=(psd_nm[3]-psd_nm[4])/(psd_nm[3]+psd_nm[4]+eps)
                    r5n=(psd_nm[6]-psd_nm[7])/(psd_nm[6]+psd_nm[7]+eps)
                    feat_nm = np.concatenate([r1n,psd_nm[0],psd_nm[5],r4n,r5n])
                    if feat_nm.shape[0]!=645: continue
                    X_all.append(feat_nm.astype(np.float32)); y_all.append(0)
            except: pass

        if len(X_all) >= 20:
            X = np.array(X_all); y = np.array(y_all)
            valid = np.isfinite(X).all(axis=1)
            X,y = X[valid],y[valid]
            Xs  = StandardScaler().fit_transform(X)
            lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage=0.4)
            skf = StratifiedKFold(10, shuffle=True, random_state=42)
            mw=(y==3)|(y==0); yw=np.where(y[mw]==3,1,0)
            if len(np.unique(yw))>1 and mw.sum()>=10:
                try: acw=float(np.nanmean(cross_val_score(lda,Xs[mw],yw,cv=skf,error_score=np.nan)))
                except: pass
            ms=(y!=3)
            if len(np.unique(y[ms]))>1 and ms.sum()>=10:
                try: acs=float(np.nanmean(cross_val_score(lda,Xs[ms],y[ms],cv=skf,error_score=np.nan)))
                except: pass
            mean_p = float(np.nanmean([acs,acw]))

    return {
        'chan_stats': chan_stats,
        'counts'    : counts,
        'n_total'   : len(trials),
        'acs'       : acs,
        'acw'       : acw,
        'mean'      : mean_p,
    }


# ============================================================
# ANALYSE GRAPHIQUE PAR SUJET
# ============================================================

def plot_sujet(subj_id, days_data, out_dir):
    """
    Génère un graphique complet pour un sujet.
    days_data = liste de (label_jour, trials, stats) ou None
    """
    n_days = len(days_data)
    fig = plt.figure(figsize=(22, 20))
    fig.patch.set_facecolor('#F8F9FA')
    fig.suptitle(f"Analyse complète — {subj_id}",
                 fontsize=16, fontweight='bold', y=0.98,
                 bbox=dict(boxstyle='round,pad=0.4',
                           facecolor='#1F4E79', edgecolor='none'),
                 color='white')

    gs = gridspec.GridSpec(4, 3, figure=fig,
                           hspace=0.45, wspace=0.35,
                           top=0.93, bottom=0.05)

    colors_days = ['#E74C3C', '#2980B9', '#27AE60']
    jour_labels = [d[0] for d in days_data]

    # ——————————————————————————————————
    # PLOT 1-3 — Signal brut canaux critiques par jour
    # ——————————————————————————————————
    for di, (jour_label, trials, stats) in enumerate(days_data):
        ax = fig.add_subplot(gs[0, di])
        ax.set_facecolor('#FAFAFA')
        if trials is None or len(trials) == 0:
            ax.text(0.5, 0.5, 'MANQUANT', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14, color='gray')
            ax.set_title(f"{jour_label}", fontsize=11, fontweight='bold')
            continue

        # Prendre un trial au hasard
        idx = np.random.randint(len(trials))
        raw_ch, label, _ = trials[idx]
        t = np.arange(raw_ch.shape[1]) / FS

        crit_colors = {'C3':'#E74C3C', 'C4':'#27AE60', 'Cz':'#2980B9'}
        for ci, cname in enumerate(CHAN_NAMES):
            if cname in CRITICAL:
                sig = raw_ch[ci] - raw_ch[ci].mean()
                ax.plot(t, sig, linewidth=0.8,
                        color=crit_colors.get(cname,'gray'),
                        label=cname, alpha=0.9)

        is_test = trials[idx][2]
        fmt = "Test" if is_test else "Training"
        lbl_names = {0:'IDLE', 1:'RIGHT', 2:'LEFT', 3:'WALK'}
        ax.set_title(f"{jour_label} [{fmt}]\nTrial {label} = {lbl_names.get(label,'?')}",
                     fontsize=10, fontweight='bold')
        ax.set_xlabel("Temps (s)", fontsize=9)
        ax.set_ylabel("Amplitude (µV)", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # ——————————————————————————————————
    # PLOT 4-6 — PSD canaux critiques par jour
    # ——————————————————————————————————
    for di, (jour_label, trials, stats) in enumerate(days_data):
        ax = fig.add_subplot(gs[1, di])
        ax.set_facecolor('#FAFAFA')
        if trials is None or len(trials) == 0:
            ax.text(0.5, 0.5, 'MANQUANT', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14, color='gray')
            continue

        crit_colors = {'C3':'#E74C3C', 'C4':'#27AE60', 'Cz':'#2980B9'}
        for ci, cname in enumerate(CHAN_NAMES):
            if cname in CRITICAL:
                # Moyenne PSD sur tous les trials actifs
                psds = []
                for raw_ch, label, _ in trials:
                    if label not in [1,2,3]: continue
                    sig = raw_ch[ci, 1050:1306]
                    if len(sig) < 256: continue
                    f, p = welch(sig - sig.mean(), fs=FS, nperseg=min(256,len(sig)))
                    psds.append(p)
                if psds:
                    psd_mean = np.mean(psds, axis=0)
                    mask = f <= 50
                    ax.semilogy(f[mask], psd_mean[mask],
                                color=crit_colors.get(cname,'gray'),
                                linewidth=1.5, label=cname, alpha=0.9)

        ax.axvspan(8, 13, alpha=0.15, color='blue')
        ax.axvspan(13, 30, alpha=0.15, color='green')
        ax.text(10.5, ax.get_ylim()[1] if not ax.get_ylim()[1]==1.0 else 1,
                'Mu', ha='center', fontsize=7, color='blue', fontweight='bold')
        ax.text(21, ax.get_ylim()[1] if not ax.get_ylim()[1]==1.0 else 1,
                'Beta', ha='center', fontsize=7, color='green', fontweight='bold')
        ax.set_title(f"PSD — {jour_label}", fontsize=10, fontweight='bold')
        ax.set_xlabel("Fréquence (Hz)", fontsize=9)
        ax.set_ylabel("PSD (µV²/Hz)", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # ——————————————————————————————————
    # PLOT 7 — Barres std par canal (tous jours)
    # ——————————————————————————————————
    ax7 = fig.add_subplot(gs[2, :2])
    ax7.set_facecolor('#FAFAFA')
    x = np.arange(len(CHAN_NAMES))
    width = 0.25
    for di, (jour_label, trials, stats) in enumerate(days_data):
        if stats is None: continue
        stds = [s['std'] for s in stats['chan_stats']]
        bars = ax7.bar(x + di*width, stds, width,
                       label=jour_label,
                       color=colors_days[di], alpha=0.8,
                       edgecolor='white', linewidth=0.5)
    ax7.axhline(y=10, color='green', linestyle='--',
                linewidth=1.5, label='Seuil BON (10µV)')
    ax7.axhline(y=5, color='orange', linestyle='--',
                linewidth=1.5, label='Seuil MIN (5µV)')
    ax7.set_title("Écart-type par canal et par jour (µV)",
                  fontsize=11, fontweight='bold')
    ax7.set_xlabel("Canal", fontsize=10)
    ax7.set_ylabel("STD (µV)", fontsize=10)
    ax7.set_xticks(x + width)
    ax7.set_xticklabels(CHAN_NAMES, fontsize=9)
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.2, axis='y')
    ax7.spines['top'].set_visible(False)
    ax7.spines['right'].set_visible(False)

    # ——————————————————————————————————
    # PLOT 8 — Distribution des classes
    # ——————————————————————————————————
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.set_facecolor('#FAFAFA')
    classes = ['RIGHT\n(1)', 'LEFT\n(2)', 'WALK\n(3)', 'IDLE\n(0)']
    class_keys = [1, 2, 3, 0]
    x_cls = np.arange(len(classes))
    width_cls = 0.25
    for di, (jour_label, trials, stats) in enumerate(days_data):
        if stats is None: continue
        vals = [stats['counts'].get(k, 0) for k in class_keys]
        ax8.bar(x_cls + di*width_cls, vals, width_cls,
                label=jour_label, color=colors_days[di],
                alpha=0.8, edgecolor='white')
    ax8.set_title("Distribution des classes par jour",
                  fontsize=11, fontweight='bold')
    ax8.set_xlabel("Classe", fontsize=10)
    ax8.set_ylabel("Nombre de trials", fontsize=10)
    ax8.set_xticks(x_cls + width_cls)
    ax8.set_xticklabels(classes, fontsize=9)
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.2, axis='y')
    ax8.spines['top'].set_visible(False)
    ax8.spines['right'].set_visible(False)

    # ——————————————————————————————————
    # PLOT 9 — Progression RLDA
    # ——————————————————————————————————
    ax9 = fig.add_subplot(gs[3, :])
    ax9.set_facecolor('#FAFAFA')

    jours_ok    = []
    steps_vals  = []
    walk_vals   = []
    mean_vals   = []

    for jour_label, trials, stats in days_data:
        if stats is None or np.isnan(stats['mean']): continue
        jours_ok.append(jour_label)
        steps_vals.append(stats['acs'])
        walk_vals.append(stats['acw'])
        mean_vals.append(stats['mean'])

    if jours_ok:
        x_rlda = np.arange(len(jours_ok))
        ax9.plot(x_rlda, steps_vals, 'o-', color='#E74C3C',
                 linewidth=2, markersize=8, label='BCI_STEPS', zorder=3)
        ax9.plot(x_rlda, walk_vals, 's-', color='#2980B9',
                 linewidth=2, markersize=8, label='BCI_WALK', zorder=3)
        ax9.plot(x_rlda, mean_vals, '^-', color='#27AE60',
                 linewidth=2.5, markersize=10, label='MEAN', zorder=4,
                 markeredgecolor='white', markeredgewidth=1.5)

        # Valeurs au-dessus des points
        for xi, (s, w, m) in enumerate(zip(steps_vals, walk_vals, mean_vals)):
            ax9.annotate(f'{m:.3f}', (xi, m),
                        textcoords="offset points", xytext=(0,10),
                        ha='center', fontsize=10, fontweight='bold',
                        color='#27AE60')

        ax9.axhline(y=0.71, color='gray', linestyle='--',
                    linewidth=1, alpha=0.7, label='Référence Alchalabi Jour 1 (~0.71)')
        ax9.set_xticks(x_rlda)
        ax9.set_xticklabels(jours_ok, fontsize=11)
        ax9.set_ylim(0.5, 1.0)
        ax9.set_title("Progression des performances RLDA par jour\n(10-fold CV, gamma=0.4, move1)",
                      fontsize=12, fontweight='bold')
        ax9.set_xlabel("Jour", fontsize=11)
        ax9.set_ylabel("Accuracy", fontsize=11)
        ax9.legend(fontsize=10)
        ax9.grid(True, alpha=0.2)
        ax9.spines['top'].set_visible(False)
        ax9.spines['right'].set_visible(False)
    else:
        ax9.text(0.5, 0.5, 'Données insuffisantes pour RLDA',
                ha='center', va='center', transform=ax9.transAxes,
                fontsize=14, color='gray')

    png_path = out_dir / f"{subj_id}_analyse_complete.png"
    plt.savefig(str(png_path), dpi=150, bbox_inches='tight',
                facecolor='#F8F9FA')
    plt.close()
    print(f"    → Graphique sauvegardé : {png_path.name}")
    return str(png_path)


# ============================================================
# GRAPHIQUE GLOBAL TOUS SUJETS
# ============================================================

def plot_global(all_results, out_dir):
    """Graphique de synthèse pour tous les sujets."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.patch.set_facecolor('#F8F9FA')
    fig.suptitle("Synthèse globale — Données EEG Bilal\nTous sujets, tous jours",
                 fontsize=16, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.4',
                           facecolor='#1F4E79', edgecolor='none'),
                 color='white')

    subj_ids = list(all_results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(subj_ids)))

    # ——————————————————————————————————
    # PLOT 1 — MEAN RLDA par sujet et jour
    # ——————————————————————————————————
    ax1 = axes[0, 0]
    ax1.set_facecolor('#FAFAFA')
    for si, subj_id in enumerate(subj_ids):
        days = all_results[subj_id]
        x_vals, y_vals = [], []
        for di, (jour_label, stats) in enumerate(days):
            if stats and not np.isnan(stats['mean']):
                x_vals.append(di + 1)
                y_vals.append(stats['mean'])
        if x_vals:
            ax1.plot(x_vals, y_vals, 'o-', color=colors[si],
                    linewidth=1.5, markersize=6,
                    label=subj_id, alpha=0.8)
    ax1.axhline(y=0.71, color='gray', linestyle='--',
                linewidth=1, label='Réf. Alchalabi')
    ax1.set_title("MEAN RLDA par sujet et par jour",
                  fontsize=11, fontweight='bold')
    ax1.set_xlabel("Jour", fontsize=10)
    ax1.set_ylabel("MEAN accuracy", fontsize=10)
    ax1.set_xticks([1, 2, 3])
    ax1.set_xticklabels(['Jour 1', 'Jour 2', 'Jour 3'])
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.2)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ——————————————————————————————————
    # PLOT 2 — BCI_STEPS vs BCI_WALK Jour 1
    # ——————————————————————————————————
    ax2 = axes[0, 1]
    ax2.set_facecolor('#FAFAFA')
    steps_j1, walk_j1, labels_j1 = [], [], []
    for subj_id in subj_ids:
        days = all_results[subj_id]
        if days and days[0][1] and not np.isnan(days[0][1]['acs']):
            steps_j1.append(days[0][1]['acs'])
            walk_j1.append(days[0][1]['acw'])
            labels_j1.append(subj_id)
    scatter = ax2.scatter(steps_j1, walk_j1, c=range(len(steps_j1)),
                          cmap='tab10', s=100, zorder=3,
                          edgecolors='white', linewidths=1.5)
    for i, label in enumerate(labels_j1):
        ax2.annotate(label, (steps_j1[i], walk_j1[i]),
                    textcoords="offset points", xytext=(5,5),
                    fontsize=8, color='#2C3E50')
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.set_title("BCI_STEPS vs BCI_WALK — Jour 1\n(chaque point = un sujet)",
                  fontsize=11, fontweight='bold')
    ax2.set_xlabel("BCI_STEPS accuracy", fontsize=10)
    ax2.set_ylabel("BCI_WALK accuracy", fontsize=10)
    ax2.grid(True, alpha=0.2)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # ——————————————————————————————————
    # PLOT 3 — Distribution trials par sujet
    # ——————————————————————————————————
    ax3 = axes[1, 0]
    ax3.set_facecolor('#FAFAFA')
    subj_names, totals_j1, totals_j2, totals_j3 = [], [], [], []
    for subj_id in subj_ids:
        days = all_results[subj_id]
        subj_names.append(subj_id)
        for di in range(3):
            val = days[di][1]['n_total'] if di < len(days) and days[di][1] else 0
            [totals_j1, totals_j2, totals_j3][di].append(val)

    x = np.arange(len(subj_names))
    w = 0.25
    ax3.bar(x - w, totals_j1, w, label='Jour 1', color='#E74C3C', alpha=0.8)
    ax3.bar(x,     totals_j2, w, label='Jour 2', color='#2980B9', alpha=0.8)
    ax3.bar(x + w, totals_j3, w, label='Jour 3', color='#27AE60', alpha=0.8)
    ax3.axhline(y=300, color='gray', linestyle='--',
                linewidth=1, label='Objectif (300 trials)')
    ax3.set_title("Nombre de trials par sujet et par jour",
                  fontsize=11, fontweight='bold')
    ax3.set_xlabel("Sujet", fontsize=10)
    ax3.set_ylabel("Nombre de trials", fontsize=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(subj_names, rotation=30, fontsize=8)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.2, axis='y')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # ——————————————————————————————————
    # PLOT 4 — Boxplot MEAN par config
    # ——————————————————————————————————
    ax4 = axes[1, 1]
    ax4.set_facecolor('#FAFAFA')
    data_box = {'Jour 1': [], 'Jour 2': [], 'Jour 3': []}
    for subj_id in subj_ids:
        days = all_results[subj_id]
        labels_box = ['Jour 1', 'Jour 2', 'Jour 3']
        for di, lbl in enumerate(labels_box):
            if di < len(days) and days[di][1] and not np.isnan(days[di][1]['mean']):
                data_box[lbl].append(days[di][1]['mean'])

    bp = ax4.boxplot([data_box[k] for k in data_box.keys()],
                     labels=list(data_box.keys()),
                     patch_artist=True,
                     medianprops={'color':'black','linewidth':2})
    colors_box = ['#E74C3C', '#2980B9', '#27AE60']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax4.axhline(y=0.71, color='gray', linestyle='--',
                linewidth=1, label='Réf. Alchalabi (~0.71)')
    ax4.set_title("Distribution MEAN RLDA par jour\n(tous sujets)",
                  fontsize=11, fontweight='bold')
    ax4.set_ylabel("MEAN accuracy", fontsize=10)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.2, axis='y')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    png_path = out_dir / "synthese_globale.png"
    plt.savefig(str(png_path), dpi=150, bbox_inches='tight',
                facecolor='#F8F9FA')
    plt.close()
    print(f"\n  → Synthèse globale sauvegardée : {png_path.name}")


# ============================================================
# RAPPORT CSV GLOBAL
# ============================================================

def save_csv(all_results, csv_path):
    """Sauvegarde un CSV global avec tous les résultats."""
    rows = []
    for subj_id, days in all_results.items():
        for jour_label, stats in days:
            if stats is None:
                rows.append({
                    'sujet': subj_id, 'jour': jour_label,
                    'status': 'MANQUANT',
                    'n_trials': 0, 'n_right': 0, 'n_left': 0,
                    'n_walk': 0, 'n_idle': 0,
                    'std_C3': '-', 'std_C4': '-', 'std_Cz': '-',
                    'bci_steps': '-', 'bci_walk': '-', 'mean': '-'
                })
                continue
            std_by_name = {s['nom']: s['std'] for s in stats['chan_stats']}
            rows.append({
                'sujet'    : subj_id,
                'jour'     : jour_label,
                'status'   : 'OK',
                'n_trials' : stats['n_total'],
                'n_right'  : stats['counts'].get(1, 0),
                'n_left'   : stats['counts'].get(2, 0),
                'n_walk'   : stats['counts'].get(3, 0),
                'n_idle'   : stats['counts'].get(0, 0),
                'std_C3'   : f"{std_by_name.get('C3', 0):.2f}",
                'std_C4'   : f"{std_by_name.get('C4', 0):.2f}",
                'std_Cz'   : f"{std_by_name.get('Cz', 0):.2f}",
                'bci_steps': f"{stats['acs']:.3f}" if not np.isnan(stats['acs']) else 'N/A',
                'bci_walk' : f"{stats['acw']:.3f}" if not np.isnan(stats['acw']) else 'N/A',
                'mean'     : f"{stats['mean']:.3f}" if not np.isnan(stats['mean']) else 'N/A',
            })

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  → CSV global sauvegardé : {csv_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str,
        default="/Users/emma/Desktop/copy/data/etude 2/")
    ap.add_argument("--out", type=str,
        default="/Users/emma/Desktop/code_RDLA/analyse_bilal/")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("ANALYSE COMPLÈTE — Données EEG Bilal")
    print("Quantitative + Qualitative | Tous sujets | Tous jours")
    print("="*70)

    # Chargement filtre IIR
    sos = load_iir()
    if sos is None:
        print("⚠️  Filtre IIR non trouvé — RLDA désactivé")

    all_results = {}

    for subj_id, jour_names in SUBJECTS.items():
        print(f"\n{'='*60}")
        print(f"SUJET : {subj_id}")
        print(f"{'='*60}")

        days_data   = []  # pour graphique
        days_results = []  # pour CSV

        for d, jour_name in enumerate(jour_names):
            jour_label = f"Jour {d+1}"
            folder     = Path(args.base) / subj_id / jour_name

            if not folder.exists():
                print(f"  {jour_label} : [MANQUANT]")
                days_data.append((jour_label, None, None))
                days_results.append((jour_label, None))
                continue

            print(f"  {jour_label} : chargement...", end="", flush=True)
            trials, n_runs, is_test = load_day_raw(folder)

            if not trials:
                print(f" [VIDE]")
                days_data.append((jour_label, None, None))
                days_results.append((jour_label, None))
                continue

            labels = np.array([t[1] for t in trials])
            print(f" {len(trials)} trials "
                  f"(R={(labels==1).sum()} L={(labels==2).sum()} "
                  f"W={(labels==3).sum()} I={(labels==0).sum()})")

            print(f"    → Analyse quantitative...", end="", flush=True)
            stats = analyse_quantitative(trials, sos)
            if stats:
                acs_str = f"{stats['acs']:.3f}" if not np.isnan(stats['acs']) else "N/A"
                acw_str = f"{stats['acw']:.3f}" if not np.isnan(stats['acw']) else "N/A"
                mn_str  = f"{stats['mean']:.3f}" if not np.isnan(stats['mean']) else "N/A"
                print(f" STEPS={acs_str} WALK={acw_str} MEAN={mn_str}")

                # Affichage détaillé stats signal
                print(f"    {'Canal':<8} {'Std(µV)':>10} {'Max(µV)':>10} {'Artefacts':>10} {'%Valide':>10} {'Qualité'}")
                print(f"    {'-'*60}")
                for s in stats['chan_stats']:
                    marker = "⭐" if s['is_critical'] else "  "
                    qual = "✅ BON" if s['std'] >= 10 else ("⚠️  MOYEN" if s['std'] >= 5 else "❌ FAIBLE")
                    print(f"    {marker}{s['nom']:<8} {s['std']:>10.2f} {s['max']:>10.2f} {s['n_art']:>10} {s['pct_valid']:>9.1f}% {qual}")
                
                n_right = stats['counts'].get(1, 0)
                n_left  = stats['counts'].get(2, 0)
                n_walk  = stats['counts'].get(3, 0)
                n_idle  = stats['counts'].get(0, 0)
                print(f"    Distribution : RIGHT={n_right} LEFT={n_left} WALK={n_walk} IDLE={n_idle}")
                print(f"    RLDA         : BCI_STEPS={acs_str} | BCI_WALK={acw_str} | MEAN={mn_str}")
                print()

            days_data.append((jour_label, trials, stats))
            days_results.append((jour_label, stats))

        # Graphique par sujet
        print(f"  → Génération graphique...", end="", flush=True)
        plot_sujet(subj_id, days_data, out_dir)

        all_results[subj_id] = days_results

    # Graphique global
    print(f"\n{'='*60}")
    print("GÉNÉRATION SYNTHÈSE GLOBALE")
    plot_global(all_results, out_dir)

    # CSV global
    csv_path = out_dir / "analyse_complete_bilal.csv"
    save_csv(all_results, str(csv_path))

    # Résumé terminal
    print(f"\n{'='*70}")
    print("RÉSUMÉ GLOBAL")
    print(f"{'='*70}")
    print(f"\n{'Sujet':<10} {'Jour 1':>10} {'Jour 2':>10} {'Jour 3':>10}")
    print("-"*44)
    for subj_id, days in all_results.items():
        vals = []
        for _, stats in days:
            if stats and not np.isnan(stats['mean']):
                vals.append(f"{stats['mean']:.3f}")
            else:
                vals.append("N/A")
        while len(vals) < 3:
            vals.append("N/A")
        print(f"  {subj_id:<10} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10}")

    print(f"\n✅ Tous les fichiers sauvegardés dans : {out_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
