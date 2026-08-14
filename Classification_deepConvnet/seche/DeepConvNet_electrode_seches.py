"""
DeepConvNet — PyTorch CPU (version corrigée)
==============================================
Implémentation de DeepConvNet pour la classification EEG.
Schirrmeister et al. (2017) "Deep learning with convolutional neural
networks for EEG decoding and visualization"

CORRECTIONS APPORTÉES :
1. BASE_FOLDER paramétrable via --subject
2. Support de fichiers plats runN[_suffixe].csv / runNunity[_suffixe].csv
3. parse_time gère le format AM/PM ('1:24:43 PM')
4. Synchronisation EEG/Unity par horloge absolue (au lieu de t=0 supposé)
5. Suppression des chiffres comparatifs codés en dur (RLDA/EEGNet obsolètes)

Architecture :
  Block 0 : Conv temporelle (1, 25) + Conv spatiale (n_ch, 1)
  Block 1 : Conv (1, 10) + MaxPool + Dropout
  Block 2 : Conv (1, 10) + MaxPool + Dropout
  Block 3 : Conv (1, 10) + MaxPool + Dropout
  Classifier : Dense → n_classes

Data Augmentation :
  - Gaussian noise (3 copies)
  - Time shifting (2 copies)
  → ×6 epochs entraînement

Usage :
    python deepconvnet.py --subject 01
    python deepconvnet.py --subject 99

Prérequis :
    pip install torch numpy pandas matplotlib scipy scikit-learn

------------------------------------------------------------------------
GUIDE DE LECTURE RAPIDE (pour un futur utilisateur) :
------------------------------------------------------------------------
Ce script est le 3e pipeline de classification EEG du projet, avec la
même logique générale que les scripts RLDA et EEGNet, mais avec un
réseau de neurones plus profond : DeepConvNet (Schirrmeister et al. 2017).

Où se situe DeepConvNet par rapport aux deux autres approches :
  - RLDA     : features "faites main" (PSD, CSP, asymétrie) + LDA linéaire.
               Rapide, interprétable, mais limité si les relations dans
               le signal sont complexes/non-linéaires.
  - EEGNet   : petit réseau de neurones (peu de paramètres), conçu
               spécifiquement pour l'EEG (convolutions temporelle +
               spatiale + séparable). Bon compromis taille/performance.
  - DeepConvNet (ce script) : réseau plus profond et plus large (4 blocs
               de convolution, jusqu'à 200 filtres), à l'origine conçu
               pour des jeux de données EEG plus volumineux. Plus de
               capacité d'apprentissage, mais aussi plus de paramètres
               à entraîner : nécessite plus de données (d'où
               l'importance de la data augmentation) et est plus sujet
               au sur-apprentissage sur de petits jeux de données.

Grandes étapes du script (identiques dans l'esprit à eegnet_final.py) :
  A. Chargement des runs (find_run_pairs, load_and_filter) : synchronise
     EEG/Unity et filtre le signal (passe-haut + notch + passe-bande).
  B. Découpage en epochs (extract_epochs) avec correction de baseline
     et normalisation z-score par epoch.
  C. Augmentation de données (augment_data) : bruit gaussien + décalage
     temporel, pour multiplier le nombre d'exemples d'entraînement.
  D. Architecture DeepConvNet (classe DeepConvNet) : 4 blocs de
     convolution (temporelle+spatiale, puis 3 blocs classiques) qui
     réduisent progressivement la résolution temporelle tout en
     augmentant le nombre de filtres, suivi d'une couche dense finale.
  E. Entraînement par fold (train_fold), avec arrêt anticipé.
  F. Validation Leave-One-Run-Out (LORO) dans main().
  G. Visualisation des résultats (plot_results, plot_training_curves).
------------------------------------------------------------------------
"""

import os
import re
import glob
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sans interface graphique (utile en script/serveur)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              classification_report, ConfusionMatrixDisplay)

# Autorise PyTorch à retomber sur le CPU si une opération MPS (GPU Mac)
# n'est pas supportée, plutôt que de planter.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

# CPU forcé explicitement : sur certains Mac, l'accélération GPU (MPS)
# provoque des segfaults avec PyTorch 2.2.2, donc on reste sur CPU par sécurité.
DEVICE = torch.device("cpu")

print(f"PyTorch version : {torch.__version__}")
print(f"Device          : {DEVICE}")

# ============================================================
# CONFIGURATION
# ============================================================
# Tous les paramètres réglables du pipeline sont regroupés ici.

# Dossier racine du projet, contenant un sous-dossier par sujet.
# ⚠️ À adapter si le script est utilisé sur un autre ordinateur.
BASE_ROOT = "/Users/emma/Desktop/Projet fin de maitrise/acquisition_donnes_kaptics"

# Valeurs par défaut — écrasées dans main() selon --subject
BASE_FOLDER = os.path.join(BASE_ROOT, "Sujet 99")
OUTPUT_DIR  = os.path.join(BASE_FOLDER, "deepconvnet_results")

FACTOR          = 0.01    # conversion ADC → µV
FS              = 250     # fréquence d'échantillonnage (Hz)
CLASSES         = ["Gauche", "Droite", "Marche"]
CLASS_MAP       = {"Gauche": 0, "Droite": 1, "Marche": 2}
N_CLASSES       = 3
N_CHANNELS      = 16      # nombre de canaux EEG utilisés en entrée du réseau

HIGHPASS        = 1.0             # passe-haut, retire la dérive lente
NOTCH_FREQS     = [50.0, 60.0]    # fréquences du bruit électrique à filtrer (50Hz Europe, 60Hz Amérique)
BP_LOW          = 8.0             # borne basse du passe-bande (bande mu)
BP_HIGH         = 30.0            # borne haute du passe-bande (bande beta)

# Fenêtre d'imagerie — basée sur la temporalité définie dans GameManager.cs
ARROW_DURATION  = 2.0   # durée d'affichage de la flèche de consigne (sec)
EPOCH_START     = ARROW_DURATION   # t+2.0s — début de la fenêtre extraite (après disparition flèche)
EPOCH_END       = 4.5              # t+4.5s — fin de la fenêtre extraite
EPOCH_DURATION  = EPOCH_END - EPOCH_START  # 2.5s
BASELINE_DUR    = 0.5      # durée utilisée pour la correction de baseline (sec)
WARMUP_TIME     = 30.0     # secondes ignorées en début d'enregistrement (signal instable)
ARTIFACT_THRESH = 100.0    # µV — seuil de rejet d'un epoch pollué par un artefact
PRE_STIM        = 0.5      # (non utilisé directement dans extract_epochs, conservé pour référence/compatibilité)
POST_STIM       = 2.0

# Hyperparamètres de l'architecture DeepConvNet — nombre de filtres à
# chaque bloc de convolution (augmente progressivement en profondeur,
# comme dans un CNN classique de type VGG)
N_FILTERS_TIME  = 25    # bloc 0 — filtres de la convolution temporelle
N_FILTERS_SPAT  = 25    # bloc 0 — filtres de la convolution spatiale
N_FILTERS_2     = 50    # bloc 1
N_FILTERS_3     = 100   # bloc 2
N_FILTERS_4     = 200   # bloc 3
POOL_SIZE       = 3     # taille du max-pooling (réduction de résolution temporelle)
POOL_STRIDE     = 3     # pas du max-pooling
DROP_RATE       = 0.5   # taux de dropout (régularisation — plus élevé que EEGNet
                         # car DeepConvNet a plus de paramètres et risque plus le sur-apprentissage)

# Paramètres d'entraînement
N_EPOCHS        = 200    # nombre maximal de passages sur les données d'entraînement
BATCH_SIZE      = 8
LR              = 1e-3   # taux d'apprentissage (learning rate)
PATIENCE        = 30     # nombre d'epochs sans amélioration avant arrêt anticipé

# Data Augmentation — génère des variantes artificielles des epochs
# d'entraînement pour compenser le nombre limité d'exemples réels
# (important ici car DeepConvNet a beaucoup de paramètres à entraîner)
AUGMENT          = True
AUG_NOISE_STD    = 0.05   # écart-type du bruit gaussien ajouté
AUG_SHIFT_MAX    = 25     # décalage temporel maximal (en échantillons)
AUG_COPIES_NOISE = 3      # nombre de copies bruitées générées par epoch
AUG_COPIES_SHIFT = 2      # nombre de copies décalées générées par epoch

# ============================================================
# RECHERCHE DES FICHIERS (structure plate runN[_suffixe].csv)
# ============================================================

def find_run_pairs(base_folder):
    """
    Trouve les paires (fichier EEG, fichier Unity) pour chaque run,
    à partir de fichiers plats runN[_suffixe].csv / runNunity[_suffixe].csv.
    Retourne une liste de tuples (numéro_run, chemin_eeg, chemin_unity).
    Les runs sans fichier unity correspondant sont ignorés (message [SKIP]).
    """
    all_csvs = glob.glob(os.path.join(base_folder, "run*.csv"))

    eeg_files = {}     # {numéro_run: chemin du fichier EEG}
    unity_files = {}   # {numéro_run: chemin du fichier de marqueurs Unity}

    for f in all_csvs:
        fname = os.path.basename(f)
        # Distingue "runN.csv" (EEG) de "runNunity.csv" (marqueurs) via une regex
        m = re.match(r"run(\d+)(unity)?(_.*)?\.csv", fname, re.IGNORECASE)
        if not m:
            continue
        run_num = int(m.group(1))
        is_unity = m.group(2) is not None

        if is_unity:
            unity_files[run_num] = f
        else:
            eeg_files[run_num] = f

    pairs = []
    for run_num in sorted(eeg_files.keys()):
        if run_num in unity_files:
            pairs.append((run_num, eeg_files[run_num], unity_files[run_num]))
        else:
            print(f"  [SKIP] run{run_num} — fichier unity manquant")

    return pairs


# ============================================================
# SYNCHRONISATION CORRIGÉE
# ============================================================
# Le casque EEG et Unity ont chacun leur propre horloge ; on convertit
# tout en "secondes depuis minuit" (heure du jour) pour les aligner.

def parse_time_of_day(t_str):
    """Convertit un timestamp type '1:24:43 PM' ou '13:24:43' en secondes depuis minuit."""
    t_str = t_str.strip()
    # Essaie plusieurs formats possibles (avec/sans AM-PM, avec/sans millisecondes)
    for fmt in ("%I:%M:%S %p", "%H:%M:%S", "%I:%M:%S.%f %p", "%H:%M:%S.%f"):
        try:
            dt = datetime.strptime(t_str, fmt)
            return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6
        except ValueError:
            continue
    raise ValueError(f"Format de timestamp non reconnu : '{t_str}'")


def eeg_timestamp_to_time_of_day(unix_ts):
    """Convertit un TimeStamp Unix (secondes) en secondes depuis minuit, heure locale."""
    dt = datetime.fromtimestamp(unix_ts)
    return dt.hour * 3600 + dt.minute * 60 + dt.second


# ============================================================
# CHARGEMENT ET FILTRAGE
# ============================================================

def load_and_filter(eeg_path, unity_path, run_name):
    """
    Charge le fichier EEG brut + le fichier Unity, les synchronise via
    l'horloge absolue, puis applique le pipeline de filtrage
    (passe-haut + notch + passe-bande).

    Retourne (eeg, markers, data) — voir eegnet_final.py pour le détail,
    le fonctionnement est identique. Retourne None si le run est trop
    court (< 10 secondes).
    """
    eeg     = pd.read_csv(eeg_path)
    markers = pd.read_csv(unity_path)

    if len(eeg) < FS * 10:
        return None

    # ── Temps EEG basé sur l'horloge absolue ──
    eeg["tod"] = eeg["TimeStamp"].apply(eeg_timestamp_to_time_of_day)
    eeg["sample_in_sec"] = eeg.groupby("TimeStamp").cumcount()

    # Fréquence d'échantillonnage réelle (peut différer légèrement de FS=250Hz)
    counts_per_sec = eeg.groupby("TimeStamp").size()
    fs_empirique = counts_per_sec.median()

    eeg["time_abs"] = eeg["tod"] + eeg["sample_in_sec"] / fs_empirique

    # ── Temps Unity basé sur l'horloge absolue ──
    markers["time_abs"] = markers["Timestamp"].apply(parse_time_of_day)

    # ── Alignement : tout relatif au premier échantillon EEG ──
    t0 = eeg["time_abs"].iloc[0]
    eeg["time_rel"]     = eeg["time_abs"] - t0
    markers["time_rel"] = markers["time_abs"] - t0

    # Vérification de cohérence : un décalage négatif ou trop grand (>300s)
    # entre le début EEG et le premier marker indique un problème de sync.
    decalage = markers["time_rel"].iloc[0]
    print(f"    [SYNC] Fréq. empirique: {fs_empirique:.1f} Hz | "
          f"Décalage EEG→1er marker: {decalage:.1f}s")
    if decalage < 0 or decalage > 300:
        print(f"    [ATTENTION] Décalage suspect ({decalage:.1f}s)")

    eeg_cols = [c for c in eeg.columns if "Channel" in c]
    raw      = eeg[eeg_cols].values * FACTOR   # conversion ADC → µV

    # Filtrage :
    #   1. Passe-haut : retire la dérive lente
    b, a = butter(4, HIGHPASS/(FS/2), btype="high")
    data = filtfilt(b, a, raw, axis=0)
    #   2. Notch (une passe par fréquence) : retire le bruit secteur
    for f in NOTCH_FREQS:
        b, a = iirnotch(f/(FS/2), 30)
        data = filtfilt(b, a, data, axis=0)
    #   3. Passe-bande : ne garde que la bande mu/beta pertinente pour l'imagerie motrice
    b, a = butter(4, [BP_LOW/(FS/2), BP_HIGH/(FS/2)], btype="band")
    data = filtfilt(b, a, data, axis=0)

    return eeg, markers, data


def extract_epochs(eeg_df, markers_df, data):
    """
    Découpe le signal filtré en epochs autour de chaque marqueur
    d'événement, sur la fenêtre d'imagerie motrice
    [t+EPOCH_START, t+EPOCH_END]. Applique correction de baseline puis
    normalisation z-score par epoch (nécessaire pour un réseau de neurones).

    Retourne (epochs, labels).
    """
    epochs, labels = [], []
    n_warmup = 0
    n_reject = 0

    expected_samples = int(EPOCH_DURATION * FS)

    for _, row in markers_df[markers_df["Type"].isin(CLASSES)].iterrows():
        t = row["time_rel"]

        # Ignore les marqueurs trop proches du début de l'enregistrement
        if t < WARMUP_TIME:
            n_warmup += 1
            continue

        # Fenêtre d'imagerie motrice : après la disparition de la flèche
        t_start = t + EPOCH_START
        t_end   = t + EPOCH_END

        mask  = ((eeg_df["time_rel"] >= t_start) &
                 (eeg_df["time_rel"] <  t_end))
        epoch = data[mask.values, :]

        # Rejette les epochs incomplets (moins de 80% des échantillons attendus)
        if len(epoch) < int(expected_samples * 0.8):
            continue
        # Rejette les epochs pollués par un artefact (amplitude trop élevée)
        if np.max(np.abs(epoch)) > ARTIFACT_THRESH:
            n_reject += 1
            continue

        # Tronque à la taille attendue exacte, pour des epochs homogènes
        epoch = epoch[:expected_samples, :]
        # Correction de baseline : soustrait la moyenne des BASELINE_DUR
        # premières secondes de l'epoch
        bl    = np.mean(epoch[:int(BASELINE_DUR * FS), :], axis=0)
        epoch = epoch - bl

        # Normalisation z-score par epoch (moyenne 0, écart-type 1) —
        # aide le réseau à converger plus facilement.
        mu    = epoch.mean(axis=0)
        sig   = epoch.std(axis=0) + 1e-8   # évite la division par zéro
        epoch = (epoch - mu) / sig

        epochs.append(epoch)
        labels.append(CLASS_MAP[row["Type"]])

    if n_warmup > 0:
        print(f"    [warmup] {n_warmup} ignorés")
    if n_reject > 0:
        print(f"    [rejet] {n_reject} artefacts")

    return epochs, labels


def epochs_to_array(epochs, n_t):
    """
    Convertit une liste d'epochs (temps x canaux) en un tableau numpy
    4D (n_epochs, 1, n_canaux, n_temps) — Format : (N, 1, N_CHANNELS, N_TIMES),
    le format attendu par PyTorch pour une entrée de type "image à 1 canal".
    """
    arr = np.zeros((len(epochs), 1, N_CHANNELS, n_t), dtype=np.float32)
    for i, ep in enumerate(epochs):
        arr[i, 0, :, :] = ep[:n_t, :].T.astype(np.float32)
    return arr


# ============================================================
# DATA AUGMENTATION
# ============================================================
# DeepConvNet ayant beaucoup de paramètres, il a particulièrement besoin
# de données d'entraînement supplémentaires pour éviter le sur-apprentissage.
# On génère donc des variantes artificielles des epochs d'entraînement.

def augment_data(X, y):
    """
    Gaussian noise + Time shifting.
    Appliqué uniquement sur les données d'entraînement (jamais sur le test,
    pour ne pas fausser l'évaluation).

    Génère :
      - AUG_COPIES_NOISE copies avec bruit gaussien ajouté
      - AUG_COPIES_SHIFT copies avec décalage temporel
    → au total, x6 le nombre d'epochs d'entraînement (original + 3 bruitées + 2 décalées)
    """
    X_aug = [X]
    y_aug = [y]

    # --- Copies bruitées ---
    for _ in range(AUG_COPIES_NOISE):
        noise = np.random.normal(0, AUG_NOISE_STD,
                                 X.shape).astype(np.float32)
        X_aug.append(X + noise)
        y_aug.append(y)

    # --- Copies décalées dans le temps (shift circulaire) ---
    n_t = X.shape[3]
    for _ in range(AUG_COPIES_SHIFT):
        shifts    = np.random.randint(-AUG_SHIFT_MAX, AUG_SHIFT_MAX + 1,
                                      size=len(X))
        X_shifted = np.zeros_like(X)
        for i, shift in enumerate(shifts):
            if shift > 0:
                X_shifted[i, :, :, shift:] = X[i, :, :, :n_t - shift]
                X_shifted[i, :, :, :shift] = X[i, :, :, :shift]
            elif shift < 0:
                s = abs(shift)
                X_shifted[i, :, :, :n_t - s] = X[i, :, :, s:]
                X_shifted[i, :, :, n_t - s:] = X[i, :, :, n_t - s:]
            else:
                X_shifted[i] = X[i]
        X_aug.append(X_shifted.astype(np.float32))
        y_aug.append(y)

    X_out = np.concatenate(X_aug, axis=0)
    y_out = np.concatenate(y_aug, axis=0)
    # Mélange l'ordre des exemples après concaténation
    idx   = np.random.permutation(len(X_out))
    return X_out[idx], y_out[idx]


# ============================================================
# ARCHITECTURE DeepConvNet
# ============================================================
# DeepConvNet empile 4 blocs de convolution qui réduisent progressivement
# la résolution temporelle (via max-pooling) tout en augmentant le
# nombre de filtres (25 → 25 → 50 → 100 → 200), un peu comme un CNN
# classique de vision par ordinateur (type VGG), mais adapté à des
# signaux EEG (2D : canaux x temps, au lieu d'images 2D classiques).

class DeepConvNet(nn.Module):
    """
    DeepConvNet — Schirrmeister et al. (2017)

    Input  : (batch, 1, n_channels, n_times)
    Output : (batch, n_classes)
    """

    def __init__(self, n_channels, n_times, n_classes=3,
                 n_filters_time=25, n_filters_spat=25,
                 n_filters_2=50, n_filters_3=100, n_filters_4=200,
                 pool_size=3, pool_stride=3, drop_rate=0.5):
        super(DeepConvNet, self).__init__()

        self.drop_rate = drop_rate

        # --- Bloc 0 : convolution temporelle + convolution spatiale ---
        # Similaire au premier bloc d'EEGNet : d'abord un filtrage le
        # long du temps, puis une combinaison des canaux (spatial).
        self.conv_time = nn.Conv2d(
            1, n_filters_time,
            kernel_size=(1, 25),   # filtre temporel de 25 échantillons
            padding=(0, 12),
            bias=False
        )
        self.conv_spat = nn.Conv2d(
            n_filters_time, n_filters_spat,
            kernel_size=(n_channels, 1),   # combine tous les canaux en une fois
            bias=False
        )
        self.bn0   = nn.BatchNorm2d(n_filters_spat,
                                     track_running_stats=False)
        self.pool0 = nn.MaxPool2d(kernel_size=(1, pool_size),
                                   stride=(1, pool_stride))
        self.drop0 = nn.Dropout(drop_rate)

        # --- Bloc 1 : convolution temporelle classique ---
        self.conv1 = nn.Conv2d(
            n_filters_spat, n_filters_2,
            kernel_size=(1, 10),
            padding=(0, 5),
            bias=False
        )
        self.bn1   = nn.BatchNorm2d(n_filters_2,
                                     track_running_stats=False)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, pool_size),
                                   stride=(1, pool_stride))
        self.drop1 = nn.Dropout(drop_rate)

        # --- Bloc 2 ---
        self.conv2 = nn.Conv2d(
            n_filters_2, n_filters_3,
            kernel_size=(1, 10),
            padding=(0, 5),
            bias=False
        )
        self.bn2   = nn.BatchNorm2d(n_filters_3,
                                     track_running_stats=False)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, pool_size),
                                   stride=(1, pool_stride))
        self.drop2 = nn.Dropout(drop_rate)

        # --- Bloc 3 ---
        self.conv3 = nn.Conv2d(
            n_filters_3, n_filters_4,
            kernel_size=(1, 10),
            padding=(0, 5),
            bias=False
        )
        self.bn3   = nn.BatchNorm2d(n_filters_4,
                                     track_running_stats=False)
        self.pool3 = nn.MaxPool2d(kernel_size=(1, pool_size),
                                   stride=(1, pool_stride))
        self.drop3 = nn.Dropout(drop_rate)

        # "Dummy forward" : passe un tenseur de zéros dans les 4 blocs
        # convolutifs pour calculer automatiquement la taille du vecteur
        # aplati en sortie (dépend de n_times), sans calcul manuel.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            dummy = self.pool0(F.elu(self.bn0(self.conv_spat(self.conv_time(dummy)))))
            dummy = self.pool1(F.elu(self.bn1(self.conv1(dummy))))
            dummy = self.pool2(F.elu(self.bn2(self.conv2(dummy))))
            dummy = self.pool3(F.elu(self.bn3(self.conv3(dummy))))
            flat  = dummy.reshape(1, -1).shape[1]
        # Couche finale : classification linéaire sur les features extraites
        self.classifier = nn.Linear(flat, n_classes)

    def forward(self, x):
        # x : (batch, 1, n_canaux, n_temps)

        # Bloc 0 : temporelle → spatiale → batchnorm → activation → pooling → dropout
        x = self.conv_time(x)
        x = self.conv_spat(x)
        x = self.bn0(x)
        x = F.elu(x)
        x = self.pool0(x)
        x = self.drop0(x)

        # Bloc 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # Bloc 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)

        # Bloc 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool3(x)
        x = self.drop3(x)

        # Aplatit et classifie (reshape plutôt que view, plus robuste sur CPU)
        x = x.reshape(x.size(0), -1)
        return self.classifier(x)


# ============================================================
# ENTRAÎNEMENT
# ============================================================
# Un "fold" correspond à un split train/test dans la validation
# Leave-One-Run-Out : un run sert de test, tous les autres d'entraînement.

def train_fold(X_train, y_train, X_test, y_test, n_t):
    """
    Entraîne DeepConvNet sur un fold LORO : entraîne le modèle sur
    (X_train, y_train), l'évalue à chaque epoch sur (X_test, y_test),
    et applique l'arrêt anticipé (early stopping) basé sur la perte
    de validation, pour éviter le sur-apprentissage.

    Retourne (y_pred, y_true, acc, history).
    """
    # Augmente les données d'ENTRAÎNEMENT uniquement
    if AUGMENT:
        X_train, y_train = augment_data(X_train, y_train)
        print(f"    [augmentation] {len(y_train)} epochs")

    # Conversion en tenseurs PyTorch (float32 obligatoire)
    X_tr_t = torch.from_numpy(np.array(X_train, dtype=np.float32))
    y_tr_t = torch.from_numpy(np.array(y_train, dtype=np.int64))
    X_te_t = torch.from_numpy(np.array(X_test,  dtype=np.float32))
    y_te_t = torch.from_numpy(np.array(y_test,  dtype=np.int64))

    tr_ds = TensorDataset(X_tr_t, y_tr_t)
    te_ds = TensorDataset(X_te_t, y_te_t)
    # num_workers=0 : évite des soucis de multiprocessing sur certains Mac
    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE,
                       shuffle=True,  num_workers=0)
    te_ld = DataLoader(te_ds, batch_size=BATCH_SIZE,
                       shuffle=False, num_workers=0)

    model = DeepConvNet(
        n_channels=N_CHANNELS, n_times=n_t, n_classes=N_CLASSES,
        n_filters_time=N_FILTERS_TIME, n_filters_spat=N_FILTERS_SPAT,
        n_filters_2=N_FILTERS_2, n_filters_3=N_FILTERS_3,
        n_filters_4=N_FILTERS_4,
        pool_size=POOL_SIZE, pool_stride=POOL_STRIDE,
        drop_rate=DROP_RATE
    ).to(DEVICE)

    # Pondération des classes inversement proportionnelle à leur fréquence,
    # pour compenser un éventuel déséquilibre entre classes.
    counts  = np.bincount(y_train, minlength=N_CLASSES).astype(np.float32)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * N_CLASSES
    crit    = nn.CrossEntropyLoss(
        weight=torch.from_numpy(weights).to(DEVICE)
    )
    opt   = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    # Réduit le taux d'apprentissage si la perte de validation stagne
    sched = optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=10, factor=0.5
    )

    best_val_loss = float("inf")
    best_weights  = None
    patience_cnt  = 0
    history       = {"train_acc": [], "val_acc": [],
                     "train_loss": [], "val_loss": []}

    for epoch in range(N_EPOCHS):
        # --- Phase d'entraînement ---
        model.train()
        tr_loss, tr_ok, tr_tot = 0.0, 0, 0
        for xb, yb in tr_ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            out  = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(yb)
            tr_ok   += (out.argmax(1) == yb).sum().item()
            tr_tot  += len(yb)

        # --- Phase de validation (sur le run de test du fold) ---
        model.eval()
        vl_loss, vl_ok, vl_tot = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in te_ld:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out     = model(xb)
                loss    = crit(out, yb)
                vl_loss += loss.item() * len(yb)
                vl_ok   += (out.argmax(1) == yb).sum().item()
                vl_tot  += len(yb)

        tr_acc       = tr_ok / tr_tot
        vl_acc       = vl_ok / vl_tot
        vl_loss_mean = vl_loss / vl_tot

        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)
        history["train_loss"].append(tr_loss / tr_tot)
        history["val_loss"].append(vl_loss_mean)

        sched.step(vl_loss_mean)

        # Sauvegarde les poids si la perte de validation s'améliore ;
        # sinon incrémente le compteur de patience pour l'arrêt anticipé.
        if vl_loss_mean < best_val_loss:
            best_val_loss = vl_loss_mean
            best_weights  = {k: v.clone()
                             for k, v in model.state_dict().items()}
            patience_cnt  = 0
        else:
            patience_cnt += 1

        if patience_cnt >= PATIENCE:
            print(f"    Early stop à l'epoch {epoch + 1}")
            break

        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1:3d} | "
                  f"Train {tr_acc*100:.1f}% | "
                  f"Val   {vl_acc*100:.1f}%")

    # Recharge les meilleurs poids trouvés (pas forcément ceux de la
    # dernière epoch, si l'entraînement a continué après le meilleur point)
    if best_weights:
        model.load_state_dict(best_weights)

    # Évaluation finale sur le run de test avec le meilleur modèle
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for xb, yb in te_ld:
            xb = xb.to(DEVICE)
            all_pred.extend(model(xb).argmax(1).cpu().numpy().tolist())
            all_true.extend(yb.numpy().tolist())

    acc = accuracy_score(all_true, all_pred)
    return np.array(all_pred), np.array(all_true), acc, history


# ============================================================
# VISUALISATION
# ============================================================

def plot_results(results, all_y_true, all_y_pred,
                 histories, output_dir, subject_name):
    """
    Génère une figure récapitulative avec :
      - la matrice de confusion globale (tous runs confondus)
      - l'accuracy par run de test
      - l'accuracy par classe (Gauche/Droite/Marche)
      - les matrices de confusion détaillées des 3 meilleurs runs
    """
    chance     = 100 / N_CLASSES
    global_acc = accuracy_score(all_y_true, all_y_pred) * 100

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(
        f"DeepConvNet — Résultats LORO — Accuracy : {global_acc:.1f}%\n"
        f"Sujet : {subject_name}",
        fontsize=13, fontweight="bold"
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # --- Matrice de confusion globale ---
    ax1 = fig.add_subplot(gs[0, 0])
    cm  = confusion_matrix(all_y_true, all_y_pred)
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(
        ax=ax1, colorbar=False, cmap="Oranges"
    )
    ax1.set_title(f"Confusion globale\n{global_acc:.1f}%", fontsize=10)

    # --- Accuracy par run (barres colorées selon la performance) ---
    ax2 = fig.add_subplot(gs[0, 1])
    rnames = [r["run"] for r in results]
    accs   = [r["acc"]*100 for r in results]
    colors = ["#4CAF50" if a > chance+10 else
              "#FF9800" if a > chance else
              "#F44336" for a in accs]
    bars = ax2.bar(rnames, accs, color=colors, alpha=0.85, width=0.6)
    ax2.axhline(chance, linestyle="--", color="gray",
                linewidth=1.5, label=f"Hasard ({chance:.0f}%)")
    ax2.axhline(70, linestyle=":", color="green",
                linewidth=1, label="Bonne perf. (70%)")
    for bar, acc in zip(bars, accs):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 1,
                 f"{acc:.0f}%", ha="center",
                 fontsize=7, fontweight="bold")
    ax2.set_ylim(0, 108)
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy par run", fontsize=10)
    ax2.tick_params(axis="x", rotation=45)
    ax2.legend(fontsize=8)

    # --- Accuracy par classe (tous runs confondus) ---
    ax3 = fig.add_subplot(gs[0, 2])
    cls_colors = ["#F44336", "#FF9800", "#4CAF50"]
    for ci, (cls_name, col) in enumerate(zip(CLASSES, cls_colors)):
        mask    = np.array(all_y_true) == ci
        cls_acc = accuracy_score(
            np.array(all_y_true)[mask],
            np.array(all_y_pred)[mask]
        ) * 100
        ax3.bar(cls_name, cls_acc, color=col, alpha=0.85)
        ax3.text(ci, cls_acc + 1, f"{cls_acc:.1f}%",
                 ha="center", fontsize=9, fontweight="bold")
    ax3.axhline(chance, linestyle="--", color="gray", linewidth=1)
    ax3.set_ylabel("Accuracy (%)")
    ax3.set_title("Accuracy par classe", fontsize=10)
    ax3.set_ylim(0, 108)

    # --- Matrices de confusion détaillées des 3 meilleurs runs ---
    best_3 = sorted(results, key=lambda r: r["acc"], reverse=True)[:3]
    for i, r in enumerate(best_3):
        ax = fig.add_subplot(gs[1, i])
        cm  = confusion_matrix(r["y_true"], r["y_pred"],
                               labels=list(range(N_CLASSES)))
        ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(
            ax=ax, colorbar=False, cmap="Oranges"
        )
        ax.set_title(f"{r['run']} — {r['acc']*100:.1f}%", fontsize=10)

    plt.tight_layout()
    out = os.path.join(output_dir, "deepconvnet_resultats.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Résultats : {out}")


def plot_training_curves(histories, output_dir):
    """
    Génère une grille de sous-graphiques (un par run de test), montrant
    l'évolution de l'accuracy train/validation au fil des epochs
    d'entraînement. Utile pour repérer le sur-apprentissage.
    """
    n    = len(histories)
    cols = min(5, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    fig.suptitle("DeepConvNet — Courbes d'entraînement LORO", fontsize=12, fontweight="bold")
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    for i, (rname, hist) in enumerate(histories.items()):
        ax  = axes_flat[i]
        eps = range(1, len(hist["train_acc"]) + 1)
        ax.plot(eps, [a*100 for a in hist["train_acc"]], "#2196F3", linewidth=1.5, label="Train")
        ax.plot(eps, [a*100 for a in hist["val_acc"]],   "#F44336", linewidth=1.5, label="Val")
        ax.axhline(33.3, linestyle="--", color="gray", linewidth=1, alpha=0.6)  # niveau du hasard (3 classes)
        ax.set_title(f"Test : {rname}", fontsize=9)
        ax.set_ylabel("Acc (%)")
        ax.set_xlabel("Epoch")
        ax.set_ylim(0, 105)
        ax.legend(fontsize=7)
    # Masque les sous-graphiques vides si le nombre de runs ne remplit
    # pas exactement la grille
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.tight_layout()
    out = os.path.join(output_dir, "deepconvnet_training_curves.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  -> Courbes : {out}")


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Point d'entrée du script. Orchestre tout le pipeline :
    1. Lecture des arguments (--subject)
    2. Recherche et chargement des runs disponibles
    3. Extraction des epochs pour chaque run
    4. Conversion en tableaux numpy adaptés à PyTorch
    5. Entraînement + évaluation DeepConvNet en validation Leave-One-Run-Out
    6. Génération des figures de résultats et résumé final
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default="99",
                        help="Nom du sujet (ex: 01, 02, 99...) — données réelles")
    args = parser.parse_args()

    # Construit le chemin du dossier du sujet à partir de --subject
    global BASE_FOLDER, OUTPUT_DIR
    BASE_FOLDER = os.path.join(BASE_ROOT, f"Sujet {args.subject}")
    OUTPUT_DIR  = os.path.join(BASE_FOLDER, "deepconvnet_results")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("="*65)
    print("  DeepConvNet — PyTorch CPU")
    print("="*65)
    print(f"  Sujet   : {args.subject}")
    print(f"  Filtres : {N_FILTERS_TIME}→{N_FILTERS_SPAT}→"
          f"{N_FILTERS_2}→{N_FILTERS_3}→{N_FILTERS_4}")
    print(f"  Pool    : size={POOL_SIZE}, stride={POOL_STRIDE}")
    print(f"  Drop    : {DROP_RATE}")
    print(f"  Train   : {N_EPOCHS} epochs max, "
          f"patience={PATIENCE}, batch={BATCH_SIZE}, LR={LR}")
    print(f"  Augment : noise×{AUG_COPIES_NOISE} + "
          f"shift×{AUG_COPIES_SHIFT} → ×6")
    print(f"  Dossier : {BASE_FOLDER}\n")

    # Étape 1 : trouve tous les runs disponibles pour ce sujet
    pairs = find_run_pairs(BASE_FOLDER)
    if not pairs:
        print("[ERREUR] Aucune paire run/unity trouvée.")
        return

    # Étape 2 : charge, filtre et découpe en epochs chaque run
    all_runs = []
    for run_num, eeg_path, unity_path in pairs:
        rname = f"run{run_num}"
        print(f"  {rname}:")
        res = load_and_filter(eeg_path, unity_path, rname)
        if res is None:
            print(f"    [SKIP] données insuffisantes")
            continue
        eeg_df, markers_df, data = res
        epochs, labels = extract_epochs(eeg_df, markers_df, data)
        if not epochs:
            continue
        print(f"    {rname}: {len(epochs)} epochs  "
              f"G:{labels.count(0)} "
              f"D:{labels.count(1)} "
              f"M:{labels.count(2)}")
        all_runs.append({
            "run": rname, "epochs": epochs, "labels": labels
        })

    # Il faut au moins 2 runs valides pour faire du Leave-One-Run-Out
    if len(all_runs) < 2:
        print("[ERREUR] Pas assez de runs.")
        return

    # Toutes les epochs doivent avoir la même longueur pour former un
    # tableau numpy rectangulaire ; on prend la longueur minimale trouvée.
    n_t = min(len(ep) for r in all_runs for ep in r["epochs"])
    print(f"\n  Taille epochs : {n_t} samples ({n_t/FS:.2f}s)")

    # Étape 3 : conversion des epochs en tableaux numpy float32
    print("  Conversion numpy float32...")
    for r in all_runs:
        r["X"] = epochs_to_array(r["epochs"], n_t)
        r["y"] = np.array(r["labels"], dtype=np.int64)

    results    = []
    histories  = {}
    all_y_true = []
    all_y_pred = []

    print("\n" + "="*65)
    print("  DeepConvNet — LEAVE-ONE-RUN-OUT")
    print("="*65)

    # Étape 4 : boucle LORO — un run sert de test, les autres d'entraînement,
    # et on répète pour chaque run.
    for test_idx, test_run in enumerate(all_runs):
        train_runs = [r for i, r in enumerate(all_runs) if i != test_idx]

        X_train = np.concatenate([r["X"] for r in train_runs], axis=0)
        y_train = np.concatenate([r["y"] for r in train_runs], axis=0)
        X_test  = test_run["X"]
        y_test  = test_run["y"]

        if len(np.unique(y_train)) < 2:
            continue

        train_names = [r["run"] for r in train_runs]
        print(f"\n  Test  : {test_run['run']}")
        print(f"  Train : {', '.join(train_names)}")
        print(f"  Train : {len(y_train)} epochs | Test : {len(y_test)}")

        # Graine fixe pour la reproductibilité des résultats
        torch.manual_seed(42)
        np.random.seed(42)

        y_pred, y_true, acc, history = train_fold(
            X_train, y_train, X_test, y_test, n_t
        )

        print(f"  Accuracy : {acc*100:.1f}%  "
              f"({int(acc*len(y_true))}/{len(y_true)})")
        for ci, cn in enumerate(CLASSES):
            mask = y_true == ci
            if mask.any():
                print(f"    {cn:<10}: "
                      f"{accuracy_score(y_true[mask], y_pred[mask])*100:.1f}%"
                      f"  (n={mask.sum()})")

        results.append({
            "run": test_run["run"], "acc": acc,
            "y_true": y_true, "y_pred": y_pred
        })
        histories[test_run["run"]] = history
        all_y_true.extend(y_true.tolist())
        all_y_pred.extend(y_pred.tolist())

    if not results:
        print("[ERREUR] Aucun résultat.")
        return

    # Étape 5 : résumé global sur tous les runs
    global_acc = accuracy_score(all_y_true, all_y_pred)
    chance     = 1.0 / N_CLASSES

    print("\n" + "="*65)
    print(f"  ACCURACY GLOBALE : {global_acc*100:.1f}%")
    print(f"  Niveau de chance : {chance*100:.1f}%")
    print(f"  Delta hasard     : {(global_acc-chance)*100:+.1f}%")
    print("="*65)
    print("\n  Rapport de classification :")
    print(classification_report(
        all_y_true, all_y_pred,
        target_names=CLASSES, zero_division=0
    ))

    # Étape 6 : génération des figures
    print("-- Génération figures --")
    plot_results(results, all_y_true, all_y_pred, histories, OUTPUT_DIR, args.subject)
    plot_training_curves(histories, OUTPUT_DIR)

    print("\n" + "="*65)
    print("  RÉSUMÉ FINAL")
    print("="*65)
    print(f"  Sujet : {args.subject}")
    print(f"  Accuracy globale : {global_acc*100:.1f}%")
    print(f"  Niveau de hasard : {chance*100:.1f}%")
    print(f"\n  Figures dans : {OUTPUT_DIR}")
    print("="*65)


if __name__ == "__main__":
    main()