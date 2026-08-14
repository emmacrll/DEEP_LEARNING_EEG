"""
Classification EEG — RLDA (Regularized Linear Discriminant Analysis)
======================================================================
Script de classification des signaux EEG du casque Kaptics.

AMÉLIORATION APPORTÉE :
- Sélection des features les plus discriminantes (SelectKBest) avant
  la normalisation et la LDA.
- Argument --subject pour choisir facilement le sujet à classifier
  (utile pour lancer sur plusieurs sujets classiques en boucle).

Pipeline complet :
  1. Chargement et filtrage des runs sélectionnés (8–30 Hz)
  2. Extraction des epochs et rejet des artefacts
  3. Extraction des features : PSD par canal + ratios asymétriques + CSP
  4. Sélection des meilleures features (SelectKBest)
  5. Classification RLDA avec validation leave-one-run-out
  6. Affichage des résultats et matrice de confusion

Validation    : Leave-One-Run-Out (LORO)

Usage :
    python classification_rlda.py --subject Sujet01
    python classification_rlda.py --subject 99

Prérequis :
    pip install pandas numpy matplotlib scipy scikit-learn

------------------------------------------------------------------------
GUIDE DE LECTURE RAPIDE (pour un futur utilisateur) :
------------------------------------------------------------------------
Ce script prend des enregistrements EEG bruts (fichiers .csv) + les
marqueurs d'événements Unity (quel type d'essai — Gauche/Droite/Marche —
et à quel moment), et essaie de prédire la classe (Gauche/Droite/Marche)
à partir du signal EEG seul.

Étapes concrètes :
  A. On cherche les fichiers "runN.csv" (EEG) et "runNunity.csv"
     (marqueurs) dans le dossier du sujet (find_run_pairs).
  B. Pour chaque run : on charge et synchronise les deux fichiers
     (load_run), on filtre le signal (apply_filters), on découpe des
     fenêtres temporelles ("epochs") autour de chaque marqueur
     (extract_epochs).
  C. On calcule des "features" (des nombres qui résument chaque epoch :
     puissance du signal dans certaines bandes de fréquence, asymétrie
     gauche/droite, etc.) via extract_features.
  D. On entraîne un classifieur RLDA (une variante de LDA régularisée)
     en laissant à chaque fois un run de côté pour le test
     (run_loro_classification = validation "Leave-One-Run-Out").
  E. On affiche/sauvegarde les résultats (matrices de confusion,
     accuracy par run, importance des features) dans plot_results et
     plot_feature_importance.
  F. main() orchestre tout ça du début à la fin.
------------------------------------------------------------------------
"""

import os
import re
import glob
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import butter, filtfilt, iirnotch
from numpy.fft import rfft, rfftfreq
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              classification_report, ConfusionMatrixDisplay)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

# ============================================================
# CONFIGURATION
# ============================================================
# Tous les paramètres "réglables" du pipeline sont regroupés ici,
# pour éviter d'avoir à fouiller dans le code pour les modifier.

# Dossier racine du projet, contenant un sous-dossier par sujet
# (ex: ".../acquisition_donnes_kaptics/Sujet 99/").
# ⚠️ À adapter si le script est utilisé sur un autre ordinateur.
BASE_ROOT = r"/Users/emma/Desktop/Projet fin de maitrise/acquisition_donnes_kaptics"

# Valeurs par défaut — écrasées dans main() selon --subject
BASE_FOLDER = os.path.join(BASE_ROOT, "Sujet 99")          # dossier du sujet en cours
OUTPUT_DIR  = os.path.join(BASE_FOLDER, "classification_results")  # où sauvegarder les figures

FACTOR = 0.01   # conversion ADC → µV (facteur de calibration du casque Kaptics)
FS     = 250    # fréquence d'échantillonnage du signal EEG, en Hz

# Canaux "moteurs" (zone du cortex liée au mouvement), utilisés pour
# certaines features spécifiques (ratio mu/beta).
MOTOR_CHANNELS = {
    "C3": "Channel7 (C3)",
    "CZ": "Channel9 (CZ)",
    "C4": "Channel11 (C4)"
}

# Tous les canaux EEG utilisés pour construire les features complètes
# (les noms doivent correspondre exactement aux en-têtes des fichiers CSV).
ALL_CHANNELS = [
    "Channel7 (C3)", "Channel8 (C1)", "Channel9 (CZ)",
    "Channel10 (C2)", "Channel11 (C4)",
    "Channel3 (FC1)", "Channel4 (FCZ)", "Channel5 (FC2)",
    "Channel12 (CP3)", "Channel13 (CP1)", "Channel14 (CPZ)",
    "Channel15 (CP2)", "Channel16 (CP4)"
]

# Timing des epochs : fenêtre extraite autour de chaque marqueur d'événement
PRE_STIM  = 1.0   # secondes AVANT le marker, utilisées comme baseline
POST_STIM = 4.0   # secondes APRÈS le marker, pendant l'imagerie motrice

# Fenêtre d'analyse pour le calcul des features (en secondes après le marker)
# On ne prend pas tout [0, POST_STIM] pour éviter le transitoire juste après
# le marker (le cerveau met un peu de temps à réagir).
FEAT_START = 0.5  # début de la fenêtre d'analyse (évite le transitoire)
FEAT_END   = 4.0  # fin de la fenêtre d'analyse (fin de l'imagerie motrice)

# Rejet artefacts : si l'amplitude dépasse ce seuil (en µV) après filtrage,
# l'epoch est considéré comme pollué (mouvement, clignement, etc.) et rejeté.
ARTIFACT_THRESH = 100.0  # µV après filtrage

# Warmup — on ignore les toutes premières secondes de l'enregistrement,
# le temps que le signal EEG se stabilise (contact des électrodes, etc.)
WARMUP_TIME = 30.0  # secondes

# Bandes de fréquence utilisées pour calculer la puissance spectrale (PSD).
# "mu" et "beta" sont les bandes classiques associées à l'imagerie motrice.
FREQ_BANDS = {
    "mu":   (8,  13),
    "beta": (13, 30),
    "low_beta": (13, 20),
    "high_beta": (20, 30),
}

# Régularisation RLDA (lambda) — "auto" laisse scikit-learn choisir
# automatiquement la meilleure valeur via l'algorithme de Ledoit-Wolf.
SHRINKAGE = "auto"  # "auto" = algorithme de Ledoit-Wolf

# Sélection de features — combien de features garder (les plus
# discriminantes selon un test statistique) avant l'entraînement du LDA.
N_FEATURES_SELECT = 25

# Nombre de composantes CSP (Common Spatial Patterns) à calculer.
# Le CSP cherche des combinaisons linéaires de canaux qui maximisent
# la différence de variance entre les classes.
CSP_N_COMPONENTS = 12

# Classes à prédire, et leur encodage numérique (utilisé par sklearn).
CLASSES    = ["Gauche", "Droite", "Marche"]
CLASS_MAP  = {"Gauche": 0, "Droite": 1, "Marche": 2}

# ============================================================
# RECHERCHE DES FICHIERS (structure plate runN[_suffixe].csv)
# ============================================================

def find_run_pairs(base_folder):
    """
    Trouve les paires (fichier EEG, fichier Unity) pour chaque run,
    à partir de fichiers plats runN[_suffixe].csv / runNunity[_suffixe].csv.

    Exemple : "run1.csv" (EEG) est apparié avec "run1unity.csv" (marqueurs).
    Si un fichier unity correspondant est manquant, le run est ignoré
    (avec un message [SKIP]).

    Retourne une liste de tuples (numéro_run, chemin_eeg, chemin_unity),
    triée par numéro de run croissant.
    """
    # Récupère tous les fichiers "run*.csv" du dossier du sujet
    all_csvs = glob.glob(os.path.join(base_folder, "run*.csv"))

    eeg_files = {}     # {numéro_run: chemin du fichier EEG}
    unity_files = {}   # {numéro_run: chemin du fichier de marqueurs Unity}

    for f in all_csvs:
        fname = os.path.basename(f)
        # Extrait le numéro de run et détecte s'il s'agit du fichier "unity"
        # (marqueurs) ou du fichier EEG brut, via une regex sur le nom.
        m = re.match(r"run(\d+)(unity)?(_.*)?\.csv", fname, re.IGNORECASE)
        if not m:
            continue  # nom de fichier qui ne correspond pas au format attendu
        run_num = int(m.group(1))
        is_unity = m.group(2) is not None

        if is_unity:
            unity_files[run_num] = f
        else:
            eeg_files[run_num] = f

    # On ne garde que les runs pour lesquels on a À LA FOIS le fichier EEG
    # et le fichier unity correspondant.
    pairs = []
    for run_num in sorted(eeg_files.keys()):
        if run_num in unity_files:
            pairs.append((run_num, eeg_files[run_num], unity_files[run_num]))
        else:
            print(f"  [SKIP] run{run_num} — fichier unity manquant")

    return pairs


# ============================================================
# FONCTIONS UTILITAIRES — SYNCHRONISATION CORRIGÉE
# ============================================================
# Le casque EEG et Unity (le logiciel qui affiche les consignes et
# enregistre les marqueurs) ont chacun leur propre horloge. Pour aligner
# les deux enregistrements, on convertit tout en "secondes depuis minuit"
# (heure du jour), qui sert de référence commune.

def parse_time_of_day(t_str):
    """Convertit un timestamp type '1:24:43 PM' ou '13:24:43' en secondes depuis minuit."""
    t_str = t_str.strip()
    # On essaie plusieurs formats possibles (avec/sans AM-PM, avec/sans millisecondes)
    for fmt in ("%I:%M:%S %p", "%H:%M:%S", "%I:%M:%S.%f %p", "%H:%M:%S.%f"):
        try:
            dt = datetime.strptime(t_str, fmt)
            return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6
        except ValueError:
            continue  # ce format ne correspond pas, on essaie le suivant
    raise ValueError(f"Format de timestamp non reconnu : '{t_str}'")


def eeg_timestamp_to_time_of_day(unix_ts):
    """Convertit un TimeStamp Unix (secondes) en secondes depuis minuit, heure locale."""
    dt = datetime.fromtimestamp(unix_ts)
    return dt.hour * 3600 + dt.minute * 60 + dt.second


# ============================================================
# FONCTIONS DE CHARGEMENT ET FILTRAGE
# ============================================================

def load_run(eeg_path, unity_path, run_name):
    """
    Charge le fichier EEG brut + le fichier Unity,
    synchronisés via l'horloge absolue (heure du jour).

    Retourne (eeg, markers, eeg_cols, raw) où :
      - eeg      : DataFrame pandas du fichier EEG, avec colonnes de temps ajoutées
      - markers  : DataFrame pandas des marqueurs Unity, avec temps ajoutés
      - eeg_cols : liste des noms de colonnes contenant les canaux EEG
      - raw      : tableau numpy (échantillons x canaux) du signal converti en µV
    Retourne None si le run contient moins de 10 secondes de données
    (probablement un enregistrement raté/tronqué).
    """
    eeg     = pd.read_csv(eeg_path)
    markers = pd.read_csv(unity_path)

    # Sécurité : si l'enregistrement est trop court, on l'ignore
    if len(eeg) < FS * 10:
        return None

    # ── Temps EEG basé sur l'horloge absolue ──
    # "TimeStamp" est en secondes Unix (souvent la même valeur répétée pour
    # plusieurs échantillons dans la même seconde) ; on reconstruit un temps
    # continu en comptant la position de chaque échantillon dans sa seconde.
    eeg["tod"] = eeg["TimeStamp"].apply(eeg_timestamp_to_time_of_day)
    eeg["sample_in_sec"] = eeg.groupby("TimeStamp").cumcount()

    # Fréquence d'échantillonnage réelle (empirique), utile si elle diffère
    # légèrement de la valeur théorique FS=250Hz.
    counts_per_sec = eeg.groupby("TimeStamp").size()
    fs_empirique = counts_per_sec.median()

    eeg["time_abs"] = eeg["tod"] + eeg["sample_in_sec"] / fs_empirique

    # ── Temps Unity basé sur l'horloge absolue ──
    markers["time_abs"] = markers["Timestamp"].apply(parse_time_of_day)

    # ── Alignement : tout relatif au premier échantillon EEG ──
    # On prend le premier instant EEG comme "temps zéro" du run, et on
    # exprime tous les autres temps (EEG et markers) par rapport à lui.
    t0 = eeg["time_abs"].iloc[0]
    eeg["time_rel"]     = eeg["time_abs"] - t0
    markers["time_rel"] = markers["time_abs"] - t0

    # Vérification de cohérence : le décalage entre le début de l'EEG et le
    # premier marqueur devrait être raisonnable (quelques secondes à
    # quelques dizaines de secondes). Un décalage négatif ou énorme (>300s)
    # indique probablement un problème de synchronisation.
    decalage = markers["time_rel"].iloc[0]
    print(f"    [SYNC] Fréq. empirique: {fs_empirique:.1f} Hz | "
          f"Décalage EEG→1er marker: {decalage:.1f}s")
    if decalage < 0 or decalage > 300:
        print(f"    [ATTENTION] Décalage suspect ({decalage:.1f}s)")

    # Sélectionne uniquement les colonnes de canaux EEG, et convertit
    # les valeurs brutes ADC en microvolts via FACTOR.
    eeg_cols = [c for c in eeg.columns if "Channel" in c]
    raw      = eeg[eeg_cols].values * FACTOR

    return eeg, markers, eeg_cols, raw


def apply_filters(data, fs=250):
    """
    Pipeline de filtrage du signal EEG brut :
      1. Passe-haut 1Hz   : retire la dérive lente (drift) de la ligne de base
      2. Notch 60Hz        : retire le bruit électrique du réseau (Amérique du Nord)
      3. Passe-bande 8-30Hz : ne garde que les bandes mu/beta pertinentes
         pour l'imagerie motrice.
    Chaque filtre est appliqué avec filtfilt (filtrage "zéro-phase",
    aller-retour) pour éviter de décaler le signal dans le temps.
    """
    b, a = butter(4, 1.0 / (fs / 2), btype="high")
    d = filtfilt(b, a, data, axis=0)
    b, a = iirnotch(60.0 / (fs / 2), 30)
    d = filtfilt(b, a, d, axis=0)
    b, a = butter(4, [8.0 / (fs / 2), 30.0 / (fs / 2)], btype="band")
    d = filtfilt(b, a, d, axis=0)
    return d


# ============================================================
# EXTRACTION DES EPOCHS
# ============================================================

def extract_epochs(eeg_df, markers_df, filtered_data, eeg_cols):
    """
    Découpe le signal filtré en "epochs" : une fenêtre temporelle
    (de -PRE_STIM à +POST_STIM secondes) autour de chaque marqueur
    d'événement (Gauche / Droite / Marche).

    Applique aussi :
      - le rejet des epochs trop proches du début (warmup)
      - le rejet des epochs contenant un artefact (amplitude > seuil)
      - une correction de baseline (soustraction de la moyenne juste
        avant le marker, pour retirer un éventuel offset DC)

    Retourne (epochs, labels) : une liste de tableaux numpy (un par
    epoch valide) et la liste des labels de classe correspondants.
    """
    epochs = []
    labels = []

    # Ne garde que les marqueurs correspondant aux classes qu'on veut prédire
    task_markers = markers_df[markers_df["Type"].isin(CLASSES)]

    n_warmup  = 0
    n_artifact = 0

    for _, row in task_markers.iterrows():
        t_onset = row["time_rel"]
        cls     = row["Type"]

        # Ignore les marqueurs trop proches du début de l'enregistrement
        # (signal pas encore stabilisé)
        if t_onset < WARMUP_TIME:
            n_warmup += 1
            continue

        # Découpe la fenêtre temporelle [t_onset - PRE_STIM, t_onset + POST_STIM]
        t_start = t_onset - PRE_STIM
        t_end   = t_onset + POST_STIM
        mask    = (eeg_df["time_rel"] >= t_start) & (eeg_df["time_rel"] < t_end)
        epoch   = filtered_data[mask.values, :]

        # Vérifie qu'on a bien récupéré assez d'échantillons (au moins 80%
        # de la durée attendue) ; sinon l'epoch est incomplet et on l'ignore.
        expected = int((PRE_STIM + POST_STIM) * FS)
        if len(epoch) < int(expected * 0.8):
            continue

        # Rejet d'artefact : si l'amplitude max dépasse le seuil, l'epoch
        # est probablement pollué par un mouvement/clignement et est rejeté.
        if np.max(np.abs(epoch)) > ARTIFACT_THRESH:
            n_artifact += 1
            continue

        # Correction de baseline : soustrait la moyenne du signal dans la
        # seconde précédant le marker, pour centrer chaque canal sur zéro.
        bl_end   = int(PRE_STIM * FS)
        bl_start = int((PRE_STIM - 1.0) * FS)
        if bl_start >= 0 and bl_end <= len(epoch):
            baseline = np.mean(epoch[bl_start:bl_end, :], axis=0)
            epoch = epoch - baseline

        epochs.append(epoch)
        labels.append(CLASS_MAP[cls])

    if n_warmup > 0:
        print(f"    [warmup] {n_warmup} epochs ignorés (<{WARMUP_TIME}s)")
    if n_artifact > 0:
        print(f"    [artefact] {n_artifact} epochs rejetés (>{ARTIFACT_THRESH}µV)")

    return epochs, labels


# ============================================================
# EXTRACTION DES FEATURES — PSD + CSP
# ============================================================

def band_power(signal, fs, low, high, nperseg=256):
    """
    Calcule la puissance moyenne du signal dans une bande de fréquence
    [low, high] Hz, via la méthode de Welch (moyenne de plusieurs
    périodogrammes calculés sur des segments qui se chevauchent à 50%).
    """
    step  = nperseg // 2  # chevauchement de 50% entre segments successifs
    psds  = []
    for start in range(0, len(signal) - nperseg, step):
        # Fenêtre de Hanning pour réduire les fuites spectrales
        seg = signal[start:start + nperseg] * np.hanning(nperseg)
        psd = np.abs(rfft(seg)) ** 2 / (fs * nperseg)
        psds.append(psd)
    if not psds:
        return 0.0  # signal trop court pour calculer au moins un segment
    psd_mean = np.mean(psds, axis=0)
    freqs    = rfftfreq(nperseg, 1 / fs)
    mask     = (freqs >= low) & (freqs <= high)
    return float(np.mean(psd_mean[mask])) if np.any(mask) else 0.0


def compute_csp_filters(X_train, y_train, n_components=4):
    """
    Calcule les filtres CSP (Common Spatial Patterns) pour 3 classes
    en mode One-vs-Rest (une classe contre toutes les autres).

    Le CSP cherche des combinaisons linéaires des canaux EEG qui
    maximisent la variance pour une classe tout en la minimisant pour
    les autres (et inversement). C'est une technique très utilisée en
    interfaces cerveau-machine (BCI) basées sur l'imagerie motrice.

    X_train : tableau (n_epochs, n_canaux, n_échantillons)
    y_train : labels de classe correspondants

    Retourne la matrice de filtres W (à appliquer ensuite avec
    apply_csp_features), ou None si le calcul échoue pour toutes les classes.
    """
    classes = np.unique(y_train)
    n_ch    = X_train.shape[2]
    all_filters = []

    for cls in classes:
        # Sépare les epochs de la classe courante ("X_cls") du reste ("X_rest")
        X_cls  = X_train[y_train == cls]
        X_rest = X_train[y_train != cls]

        if len(X_cls) < 2 or len(X_rest) < 2:
            continue  # pas assez d'exemples pour estimer une covariance fiable

        def cov_norm(X):
            # Covariance spatiale moyenne, normalisée par sa trace
            # (pour rendre les epochs comparables entre eux malgré des
            # différences d'amplitude globale).
            covs = []
            for ep in X:
                c = ep @ ep.T
                tr = np.trace(c)
                if tr > 1e-12:
                    covs.append(c / tr)
            return np.mean(covs, axis=0) if covs else np.eye(n_ch)

        Sigma1 = cov_norm(X_cls)
        Sigma2 = cov_norm(X_rest)

        try:
            from scipy.linalg import eigh
            # Décomposition en valeurs propres généralisée : cherche les
            # directions qui maximisent le ratio de variance entre les
            # deux classes.
            eigenvalues, eigenvectors = eigh(Sigma1, Sigma1 + Sigma2)
            idx   = np.argsort(eigenvalues)[::-1]
            W_cls = eigenvectors[:, idx].T

            # On garde les filtres aux deux extrémités du spectre de
            # valeurs propres (les plus discriminants dans chaque sens).
            n_keep = max(1, n_components // 2)
            selected = np.concatenate([
                W_cls[:n_keep],
                W_cls[-n_keep:]
            ])
            all_filters.append(selected)
        except Exception:
            continue  # échec numérique (ex: matrice non inversible) — on ignore cette classe

    if not all_filters:
        return None

    W = np.vstack(all_filters)
    return W


def apply_csp_features(epochs_data, W, feat_start, feat_end):
    """
    Applique les filtres CSP (matrice W) à chaque epoch, puis calcule
    la log-variance de chaque composante spatiale filtrée.
    La log-variance des composantes CSP est la feature classique utilisée
    en aval d'un CSP pour la classification.
    """
    features = []
    for epoch in epochs_data:
        # Gère les deux orientations possibles de l'epoch
        # (temps x canaux, ou canaux x temps) selon la provenance des données.
        if epoch.ndim == 2 and epoch.shape[1] == W.shape[1]:
            seg = epoch[feat_start:feat_end, :].T
        elif epoch.ndim == 2 and epoch.shape[0] == W.shape[1]:
            seg = epoch[:, feat_start:feat_end]
        else:
            seg = epoch[feat_start:feat_end, :].T

        z = W @ seg                      # projection dans l'espace CSP
        var = np.var(z, axis=1)          # variance de chaque composante
        log_var = np.log(var + 1e-12)    # log-variance (feature finale)
        features.append(log_var)

    return np.array(features)


def extract_features(epochs, eeg_cols):
    """
    Extrait les features "classiques" (hors CSP) pour chaque epoch :
      - puissance spectrale (PSD) par canal et par bande de fréquence
      - ratios d'asymétrie entre C3 et C4 (utile en imagerie motrice
        gauche/droite, car l'activité corticale motrice est controlatérale)
      - ratio mu/beta pour les canaux moteurs C3/CZ/C4

    Les features CSP sont calculées séparément (dans run_loro_classification)
    car elles dépendent du split train/test (le CSP est entraîné uniquement
    sur les données d'entraînement, pour éviter toute fuite d'information).

    Retourne (feature_matrix, feature_names) :
      - feature_matrix : tableau (n_epochs, n_features)
      - feature_names  : liste des noms de features, dans le même ordre
        que les colonnes de feature_matrix.
    """
    # Ne garde que les canaux qui sont à la fois dans ALL_CHANNELS et
    # effectivement présents dans le fichier EEG chargé.
    use_cols = [c for c in ALL_CHANNELS if c in eeg_cols]
    col_idx  = {c: eeg_cols.index(c) for c in use_cols}

    feat_start = int(FEAT_START * FS)
    feat_end   = int(FEAT_END   * FS)

    feature_matrix = []
    feature_names  = []
    built_names    = False  # les noms de features ne sont construits qu'une fois

    for epoch in epochs:
        seg = epoch[feat_start:feat_end, :]  # ne garde que la fenêtre d'analyse
        features = []

        # --- Puissance spectrale par canal et par bande de fréquence ---
        for ch in use_cols:
            idx = col_idx[ch]
            ch_name = ch.split("(")[1].replace(")", "")  # ex: "Channel7 (C3)" -> "C3"
            for band, (lo, hi) in FREQ_BANDS.items():
                pw = band_power(seg[:, idx], FS, lo, hi)
                features.append(pw)
                if not built_names:
                    feature_names.append(f"{ch_name}_{band}")

        # --- Asymétrie C3/C4 (indicateur classique en imagerie motrice) ---
        c3_idx = col_idx.get("Channel7 (C3)")
        c4_idx = col_idx.get("Channel11 (C4)")

        if c3_idx is not None and c4_idx is not None:
            for band, (lo, hi) in FREQ_BANDS.items():
                pw_c3 = band_power(seg[:, c3_idx], FS, lo, hi)
                pw_c4 = band_power(seg[:, c4_idx], FS, lo, hi)
                denom = pw_c3 + pw_c4
                # ratio normalisé entre -1 et 1 : positif si C3 plus actif, négatif si C4
                ratio = (pw_c3 - pw_c4) / denom if denom > 1e-12 else 0.0
                features.append(ratio)
                if not built_names:
                    feature_names.append(f"asym_C3C4_{band}")

        # --- Ratio mu/beta pour chaque canal moteur (C3, CZ, C4) ---
        for ch_name, ch_col in MOTOR_CHANNELS.items():
            idx = col_idx.get(ch_col)
            if idx is not None:
                mu   = band_power(seg[:, idx], FS, 8,  13)
                beta = band_power(seg[:, idx], FS, 13, 30)
                ratio = mu / beta if beta > 1e-12 else 0.0
                features.append(ratio)
                if not built_names:
                    feature_names.append(f"mu_beta_ratio_{ch_name}")

        feature_matrix.append(features)
        built_names = True  # les noms sont fixés après le premier epoch

    return np.array(feature_matrix), feature_names


# ============================================================
# CLASSIFICATION RLDA — LEAVE-ONE-RUN-OUT
# ============================================================

def run_loro_classification(all_data):
    """
    Validation Leave-One-Run-Out (LORO) :
    pour chaque run disponible, on entraîne le modèle sur TOUS les
    autres runs, et on teste sur celui laissé de côté. On répète
    l'opération pour chaque run, ce qui donne une estimation robuste
    de la capacité du modèle à généraliser à un nouveau run (jamais vu
    pendant l'entraînement).

    Pour chaque split train/test, le pipeline complet est :
      1. Calcul des filtres CSP sur le train, application au train ET au test
      2. Concaténation features PSD/asymétrie + features CSP
      3. Sélection des N_FEATURES_SELECT meilleures features (SelectKBest)
      4. Normalisation (StandardScaler), ajustée sur le train seulement
      5. Entraînement du LDA régularisé (RLDA) sur le train
      6. Prédiction et évaluation sur le test

    all_data : liste de dicts, un par run, avec les clés
               "run", "X" (features PSD), "y" (labels), "epochs_raw"
               (epochs bruts, nécessaires pour calculer le CSP).

    Retourne (results, all_y_true, all_y_pred, global_acc) :
      - results     : liste de dicts avec les résultats détaillés par run test
      - all_y_true  : labels réels concaténés sur tous les runs test
      - all_y_pred  : labels prédits concaténés sur tous les runs test
      - global_acc  : accuracy globale (tous runs confondus)
    """
    n_runs = len(all_data)
    results = []

    all_y_true = []
    all_y_pred = []

    print("\n" + "=" * 60)
    print("  CLASSIFICATION RLDA — LEAVE-ONE-RUN-OUT")
    print(f"  Sélection features : top {N_FEATURES_SELECT} | CSP composantes : {CSP_N_COMPONENTS}")
    print("=" * 60)

    # Boucle principale : à chaque itération, un run différent sert de test
    for test_idx in range(n_runs):
        test_run  = all_data[test_idx]
        train_runs = [all_data[i] for i in range(n_runs) if i != test_idx]

        # Concatène les features PSD/asymétrie de tous les runs d'entraînement
        X_train = np.vstack([r["X"] for r in train_runs])
        y_train = np.concatenate([r["y"] for r in train_runs])
        X_test  = test_run["X"]
        y_test  = test_run["y"]

        if len(X_train) == 0 or len(X_test) == 0:
            print(f"  [SKIP] Pas assez de données pour {test_run['run']}")
            continue

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            print(f"  [SKIP] Trop peu de classes pour {test_run['run']}")
            continue

        # Récupère aussi les epochs "bruts" (non résumés en features), car
        # le CSP a besoin du signal temporel complet pour être calculé.
        epochs_train_raw = []
        y_train_raw      = []
        for r in train_runs:
            epochs_train_raw.extend(r["epochs_raw"])
            y_train_raw.extend(r["y"].tolist())
        y_train_raw = np.array(y_train_raw)

        epochs_test_raw = test_run["epochs_raw"]

        feat_start = int(FEAT_START * FS)
        feat_end   = int(FEAT_END   * FS)

        # Met les epochs en forme (n_epochs, n_canaux, n_échantillons)
        # pour le calcul du CSP.
        n_ch = epochs_train_raw[0].shape[1]
        X_train_3d = np.array([
            ep[feat_start:feat_end, :].T
            for ep in epochs_train_raw
        ])

        X_test_3d = np.array([
            ep[feat_start:feat_end, :].T
            for ep in epochs_test_raw
        ])

        # ⚠️ Important : le CSP est calculé UNIQUEMENT sur les données
        # d'entraînement (X_train_3d), jamais sur le test, pour éviter
        # toute fuite d'information (data leakage) qui gonflerait
        # artificiellement la performance.
        W = compute_csp_filters(X_train_3d, y_train_raw, n_components=CSP_N_COMPONENTS)

        if W is not None:
            # Applique les MÊMES filtres CSP (appris sur le train) au train et au test
            csp_train = apply_csp_features(epochs_train_raw, W,
                                            feat_start, feat_end)
            csp_test  = apply_csp_features(epochs_test_raw,  W,
                                            feat_start, feat_end)

            # Concatène les features PSD/asymétrie avec les features CSP
            X_train_combined = np.hstack([X_train, csp_train])
            X_test_combined  = np.hstack([X_test,  csp_test])
            print(f"  CSP : {csp_train.shape[1]} features ajoutées "
                  f"(total : {X_train_combined.shape[1]})")
        else:
            # Si le CSP échoue (ex: pas assez de données), on continue
            # uniquement avec les features PSD/asymétrie.
            X_train_combined = X_train
            X_test_combined  = X_test
            print(f"  CSP : échec — utilisation PSD seule")

        # ── Sélection des features les plus discriminantes ──
        # SelectKBest + f_classif : garde les k features ayant le plus
        # fort pouvoir discriminant (test statistique ANOVA), ajusté
        # UNIQUEMENT sur le train pour éviter la fuite de données.
        k = min(N_FEATURES_SELECT, X_train_combined.shape[1])
        selector = SelectKBest(score_func=f_classif, k=k)
        X_train_sel = selector.fit_transform(X_train_combined, y_train)
        X_test_sel  = selector.transform(X_test_combined)

        # Normalisation (moyenne 0, écart-type 1), ajustée sur le train seulement
        scaler    = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_sel)
        X_test_s  = scaler.transform(X_test_sel)

        # Entraînement du LDA régularisé (solver "eigen" requis pour
        # utiliser le paramètre shrinkage, i.e. la régularisation RLDA).
        clf = LinearDiscriminantAnalysis(solver="eigen", shrinkage=SHRINKAGE)
        clf.fit(X_train_s, y_train)

        y_pred = clf.predict(X_test_s)
        acc    = accuracy_score(y_test, y_pred)

        train_names = [r["run"] for r in train_runs]
        print(f"\n  Test  : {test_run['run']}")
        print(f"  Train : {', '.join(train_names)}")
        print(f"  Accuracy : {acc * 100:.1f}%  "
              f"({int(acc * len(y_test))}/{len(y_test)} corrects)")

        # Accuracy détaillée par classe, pour voir si une classe en
        # particulier est mal reconnue.
        for cls_idx, cls_name in enumerate(CLASSES):
            mask = y_test == cls_idx
            if np.any(mask):
                cls_acc = accuracy_score(y_test[mask], y_pred[mask])
                print(f"    {cls_name:<8}: {cls_acc * 100:.1f}%  "
                      f"(n={np.sum(mask)})")

        results.append({
            "test_run":  test_run["run"],
            "acc":       acc,
            "y_true":    y_test,
            "y_pred":    y_pred,
            "clf":       clf,
            "scaler":    scaler,
        })

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

    # Une fois tous les runs testés, calcule un résumé global
    if all_y_true:
        global_acc = accuracy_score(all_y_true, all_y_pred)
        chance     = 1.0 / len(CLASSES)

        print("\n" + "=" * 60)
        print(f"  ACCURACY GLOBALE (tous runs) : {global_acc * 100:.1f}%")
        print(f"  Niveau de chance             : {chance * 100:.1f}%")
        print(f"  Au-dessus du hasard          : "
              f"{(global_acc - chance) * 100:+.1f}%")
        print("=" * 60)
        print("\n  Rapport de classification :")
        print(classification_report(
            all_y_true, all_y_pred,
            target_names=CLASSES, zero_division=0
        ))
    else:
        global_acc = 0.0

    return results, np.array(all_y_true), np.array(all_y_pred), global_acc


# ============================================================
# VISUALISATION DES RÉSULTATS
# ============================================================

def plot_results(results, all_y_true, all_y_pred, feature_names, output_dir):
    """
    Génère une figure récapitulative (sauvegardée en PNG) avec :
      - la matrice de confusion globale (tous runs confondus)
      - l'accuracy par run de test (barres colorées selon la performance)
      - l'accuracy par classe et par run de test
      - les matrices de confusion détaillées des 3 meilleurs runs
    """

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Résultats Classification RLDA — Casque Kaptics\n"
                 f"Validation Leave-One-Run-Out",
                 fontsize=13, fontweight="bold")

    # Grille 2x3 pour organiser les différents graphiques
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # --- Sous-graphique 1 : matrice de confusion globale ---
    ax1 = fig.add_subplot(gs[0, 0])
    if len(all_y_true) > 0:
        cm   = confusion_matrix(all_y_true, all_y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=CLASSES
        )
        disp.plot(ax=ax1, colorbar=False, cmap="Blues")
        ax1.set_title("Matrice de confusion globale\n(tous runs)", fontsize=10)

    # --- Sous-graphique 2 : accuracy par run (barres colorées) ---
    ax2 = fig.add_subplot(gs[0, 1])
    run_names = [r["test_run"] for r in results]
    accs      = [r["acc"] * 100 for r in results]
    fig.set_size_inches(max(18, len(results)*1.6), 12)
    chance    = 100 / len(CLASSES)

    # Vert si nettement au-dessus du hasard, orange si un peu au-dessus,
    # rouge si au niveau du hasard ou en dessous.
    colors = ["#4CAF50" if a > chance + 10 else
              "#FF9800" if a > chance else
              "#F44336" for a in accs]

    bars = ax2.bar(run_names, accs, color=colors, alpha=0.85, width=0.5)
    ax2.axhline(chance, linestyle="--", color="gray", linewidth=1.5,
                label=f"Hasard ({chance:.0f}%)")
    ax2.axhline(70, linestyle=":", color="green", linewidth=1,
                label="Bonne perf. (70%)")

    # Affiche le pourcentage au-dessus de chaque barre
    for bar, acc in zip(bars, accs):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1,
                 f"{acc:.1f}%", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")

    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy par run (test)", fontsize=10)
    ax2.tick_params(axis="x", rotation=45)
    ax2.legend(fontsize=8)

    # --- Sous-graphique 3 : accuracy par classe et par run ---
    ax3 = fig.add_subplot(gs[0, 2])
    x      = np.arange(len(CLASSES))
    width  = 0.8 / max(len(results), 1)
    import matplotlib.cm as cm
    cmap = cm.get_cmap('tab10', len(results))
    colors_cls_dyn = [cmap(i) for i in range(len(results))]

    for i, r in enumerate(results):
        cls_accs = []
        for cls_idx in range(len(CLASSES)):
            mask = r["y_true"] == cls_idx
            if np.any(mask):
                cls_accs.append(
                    accuracy_score(r["y_true"][mask], r["y_pred"][mask]) * 100
                )
            else:
                cls_accs.append(0)
        ax3.bar(x + i * width, cls_accs, width,
                label=f"{r['test_run']}",
                color=colors_cls_dyn[i], alpha=0.85)

    ax3.axhline(chance, linestyle="--", color="gray", linewidth=1,
                label=f"Hasard ({chance:.0f}%)")
    ax3.set_xticks(x + width * len(results) / 2)
    ax3.set_xticklabels(CLASSES)
    ax3.set_ylabel("Accuracy (%)")
    ax3.set_title("Accuracy par classe et par run test", fontsize=10)
    ax3.set_ylim(0, 110)

    # --- Sous-graphiques 4-6 : matrices de confusion des 3 meilleurs runs ---
    results_sorted = sorted(results, key=lambda r: r['acc'], reverse=True)
    for i, r in enumerate(results_sorted[:3]):
        ax = fig.add_subplot(gs[1, i])
        cm   = confusion_matrix(r["y_true"], r["y_pred"],
                                labels=list(range(len(CLASSES))))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=CLASSES
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        acc_pct = r["acc"] * 100
        ax.set_title(
            f"Test : {r['test_run']} — {acc_pct:.1f}%",
            fontsize=10
        )

    plt.tight_layout()
    out = os.path.join(output_dir, "resultats_rlda.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  -> Figure sauvegardée : {out}")


def plot_feature_importance(results, feature_names, output_dir):
    """
    Génère un graphique en barres horizontales des 15 features les plus
    importantes, en moyennant la valeur absolue des coefficients LDA sur
    tous les runs de test. Permet de voir quels canaux/bandes de
    fréquence contribuent le plus à la classification.
    """
    if not results:
        return

    # Récupère les coefficients (poids) appris par le LDA pour chaque run
    all_coefs = []
    for r in results:
        clf = r["clf"]
        if hasattr(clf, "coef_"):
            all_coefs.append(np.abs(clf.coef_).mean(axis=0))

    if not all_coefs:
        return

    mean_coefs = np.mean(all_coefs, axis=0)

    # Garde les 15 features avec la plus forte importance moyenne
    top_idx = np.argsort(mean_coefs)[::-1][:15]
    n_total = len(mean_coefs)
    # Comme les features CSP n'ont pas de nom explicite dans feature_names
    # (elles sont ajoutées après coup), on complète la liste avec des noms
    # génériques "CSP_1", "CSP_2", etc.
    if len(feature_names) < n_total:
        n_csp = n_total - len(feature_names)
        feature_names_ext = list(feature_names) + [f"CSP_{i+1}" for i in range(n_csp)]
    else:
        feature_names_ext = list(feature_names)
    top_names  = [feature_names_ext[i] if i < len(feature_names_ext) else f"feat_{i}"
                  for i in top_idx]
    top_vals   = mean_coefs[top_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    # Couleur selon le canal concerné, pour repérer visuellement les
    # features liées à C3, CZ, C4, ou aux autres canaux.
    colors = ["#2196F3" if "C3" in n else
              "#9C27B0" if "CZ" in n else
              "#4CAF50" if "C4" in n else
              "#FF9800" for n in top_names]

    ax.barh(range(len(top_names)), top_vals[::-1],
            color=colors[::-1], alpha=0.85)
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Importance moyenne (|coef| LDA)")
    ax.set_title("Top features les plus discriminantes\n"
                 "(bleu=C3, violet=CZ, vert=C4, orange=autres)",
                 fontsize=11, fontweight="bold")

    plt.tight_layout()
    out = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Feature importance sauvegardée : {out}")


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Point d'entrée du script. Orchestre tout le pipeline :
    1. Lecture des arguments (--subject)
    2. Recherche des runs disponibles pour ce sujet
    3. Pour chaque run : chargement, filtrage, extraction des epochs
       et des features
    4. Classification RLDA en validation Leave-One-Run-Out
    5. Génération des figures de résultats
    6. Affichage d'un résumé final avec verdict sur la qualité du signal
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default="99",
                        help="Nom du sujet (ex: Sujet01, Sujet02, 99, 00, 01...)")
    args = parser.parse_args()

    # Construit le chemin du dossier du sujet à partir de l'argument --subject
    # (écrase les valeurs par défaut définies dans la section CONFIGURATION)
    global BASE_FOLDER, OUTPUT_DIR
    BASE_FOLDER = os.path.join(BASE_ROOT, f"Sujet {args.subject}")
    OUTPUT_DIR  = os.path.join(BASE_FOLDER, "classification_results")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  CLASSIFICATION EEG — RLDA")
    print("=" * 60)
    print(f"  Sujet : {args.subject}")
    print(f"  Validation : Leave-One-Run-Out")
    print(f"  Régularisation : {SHRINKAGE}")
    print(f"  Dossier : {BASE_FOLDER}\n")

    # Étape 1 : trouve tous les runs disponibles pour ce sujet
    pairs = find_run_pairs(BASE_FOLDER)
    if not pairs:
        print("[ERREUR] Aucune paire run/unity trouvée.")
        return

    all_data = []  # accumulera les données (features + epochs) de chaque run valide

    # Étape 2 : traite chaque run individuellement
    for run_num, eeg_path, unity_path in pairs:
        run_name = f"run{run_num}"
        print(f"-- Chargement {run_name} --")
        result = load_run(eeg_path, unity_path, run_name)
        if result is None:
            print(f"  [SKIP] Données insuffisantes")
            continue

        eeg_df, markers_df, eeg_cols, raw = result
        print(f"  {len(eeg_df)} échantillons ({len(eeg_df)/FS:.1f}s)")

        # Filtrage du signal (passe-haut + notch + passe-bande)
        filtered = apply_filters(raw)

        # Découpage en epochs autour de chaque marqueur d'événement
        epochs, labels = extract_epochs(eeg_df, markers_df, filtered, eeg_cols)

        if not epochs:
            print(f"  [SKIP] Aucun epoch valide")
            continue

        print(f"  {len(epochs)} epochs valides")
        for cls_idx, cls_name in enumerate(CLASSES):
            n = sum(1 for l in labels if l == cls_idx)
            print(f"    {cls_name}: {n} epochs")

        # Extraction des features PSD/asymétrie pour ce run
        X, feature_names = extract_features(epochs, eeg_cols)
        y = np.array(labels)

        # Il faut au moins 2 classes différentes dans le run pour pouvoir
        # entraîner/tester un classifieur dessus.
        unique_cls = np.unique(y)
        if len(unique_cls) < 2:
            print(f"  [SKIP] Moins de 2 classes présentes")
            continue

        print(f"  Features extraites : {X.shape[1]} features × {X.shape[0]} epochs")

        # Stocke à la fois les features déjà calculées (X, y) et les epochs
        # bruts (epochs_raw), nécessaires plus tard pour le calcul du CSP.
        all_data.append({
            "run":        run_name,
            "X":          X,
            "y":          y,
            "epochs_raw": epochs,
        })
        print()

    # Il faut au moins 2 runs valides pour pouvoir faire du Leave-One-Run-Out
    # (sinon il n'y a rien à laisser de côté pour le test).
    if len(all_data) < 2:
        print("[ERREUR] Pas assez de runs valides pour la validation LORO.")
        return

    # Étape 3 : classification RLDA en validation Leave-One-Run-Out
    results, all_y_true, all_y_pred, global_acc = run_loro_classification(all_data)

    if not results:
        print("[ERREUR] Aucun résultat de classification.")
        return

    # Étape 4 : génération des figures de résultats
    print("\n  Génération des figures...")
    plot_results(results, all_y_true, all_y_pred, feature_names, OUTPUT_DIR)
    plot_feature_importance(results, feature_names, OUTPUT_DIR)

    # Étape 5 : résumé final avec un verdict qualitatif sur la performance
    chance = 100 / len(CLASSES)

    print("\n" + "=" * 60)
    print("  RÉSUMÉ FINAL")
    print("=" * 60)
    print(f"  Sujet : {args.subject}")
    print(f"  Accuracy globale : {global_acc*100:.1f}%")
    print(f"  Niveau de hasard : {chance:.1f}%")

    if global_acc*100 > 70:
        verdict = "BONNE performance — signal discriminant"
    elif global_acc*100 > chance + 10:
        verdict = "PERFORMANCE MODÉRÉE — signal partiellement discriminant"
    elif global_acc*100 > chance:
        verdict = "PERFORMANCE FAIBLE — légèrement au-dessus du hasard"
    else:
        verdict = "PAS MIEUX QUE LE HASARD — données insuffisantes"

    print(f"  Verdict : {verdict}")
    print(f"\n  Figures dans : {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()