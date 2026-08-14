#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              LANCEUR EEGNET — 8 SUJETS RETENUS, 7 CONFIGURATIONS            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Ce script orchestre l'évaluation complète d'EEGNet sur les 8 sujets        ║
║  retenus après l'analyse de qualité (Subj18 exclu).                         ║
║                                                                              ║
║  Il ne contient pas le code EEGNet lui-même — il appelle eegnet_4class.py   ║
║  en sous-processus pour chaque combinaison sujet × configuration.            ║
║                                                                              ║
║  7 configurations reproduisant la Figure 5 d'Alchalabi et al. (2021) :      ║
║    Config 1     : Jour 1 seul        → mode LORO                            ║
║    Config 2     : Jour 2 seul        → mode LORO                            ║
║    Config 3     : Jour 3 seul        → mode LORO                            ║
║    Config 1+2   : Train J1, Test J2  → mode Train/Test                     ║
║    Config 1+3   : Train J1, Test J3  → mode Train/Test                     ║
║    Config 2+3   : Train J2, Test J3  → mode Train/Test                     ║
║    Config 1+2+3 : Train J1+J2, Test J3 → mode Train/Test                   ║
║                                                                              ║
║  Sujets retenus : Subj05, subj06, Subj07, Subj09,                           ║
║                   Subj10, Subj11, Subj12, Subj17                            ║
║  Subj18 exclu : anomalies signal sur 2 jours sur 3                          ║
║                                                                              ║
║  Durée estimée : ~3 min/sujet × 8 sujets × 7 configs ≈ 3 heures            ║
║                                                                              ║
║  Usage :                                                                     ║
║    python eegnet_bons_sujets.py                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import subprocess   # pour lancer eegnet_4class.py en sous-processus
import sys          # pour récupérer l'exécutable Python courant
import csv          # pour sauvegarder les résultats en CSV
import re           # pour extraire mean_bal_acc depuis la sortie texte
from pathlib import Path  # pour manipuler les chemins de fichiers


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              CONFIGURATION GLOBALE                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Les 8 sujets retenus après analyse qualité (voir selection_sujets.py)
# Subj18 est exclu car anomalies signal sévères sur Jours 1 et 3
SUJETS = ["Subj05", "subj06", "Subj07", "Subj09",
          "Subj10", "Subj11", "Subj12", "Subj17"]

# Chemin racine contenant tous les dossiers npz_all_subjects_dayX
NPZ_BASE = "/Users/emma/Desktop/code_pipeline"

# Chemin vers le script EEGNet à appeler en sous-processus
EEGNET = "/Users/emma/Desktop/code_pipeline/models/eegnet_4class.py"

# Fichier CSV de sortie avec tous les résultats
OUTPUT_CSV = "/Users/emma/Desktop/code_RDLA/eegnet_bons_sujets.csv"

# ──────────────────────────────────────────────────────────────────────────────
# NOMENCLATURE DES FICHIERS .npz
# ──────────────────────────────────────────────────────────────────────────────
# Les fichiers .npz sont nommés selon le pattern :
#   {Sujet}_{Jour}_{bande}_{suffixe}.npz
#
# Trois types de suffixes selon l'origine :
#   - motor8_guessB : Jour 1 (format Training — données initiales de calibration)
#   - auto          : Jours 2 et 3 (format Test — données d'évaluation)
#   - merged        : jours fusionnés (Jour 1+2, 1+3, etc.)
#
# Deux bandes fréquentielles :
#   - band0812 : bande MU (8-12Hz)  → entrée du GATE
#   - band1330 : bande BETA (13-30Hz) → combinée avec MU pour l'EXPERT
# ──────────────────────────────────────────────────────────────────────────────

# Définition des 7 configurations d'évaluation
# Chaque config est un dict décrivant :
#   - id          : identifiant court ("1", "1+2", etc.)
#   - label       : description lisible
#   - mode        : "loro" (mono-jour) ou "traintest" (multi-jours)
#   - Pour loro   : dossier + suffixe des fichiers MU et BETA
#   - Pour traintest : dossier + suffixe pour train ET test (MU et BETA)
CONFIGS = [
    # ── Configs mono-jour → mode LORO (Leave-One-Run-Out) ──
    {
        "id"          : "1",
        "label"       : "Jour 1 seul",
        "mode"        : "loro",  # LORO car un seul jour disponible
        "dossier_mu"  : "npz_all_subjects_day1",
        "dossier_beta": "npz_all_subjects_day1",
        "suffix_mu"   : "Jour1_band0812_motor8_guessB",  # format Training
        "suffix_beta" : "Jour1_band1330_motor8_guessB",
    },
    {
        "id"          : "2",
        "label"       : "Jour 2 seul",
        "mode"        : "loro",
        "dossier_mu"  : "npz_all_subjects_day2",
        "dossier_beta": "npz_all_subjects_day2",
        "suffix_mu"   : "Jour2_band0812_auto",  # format Test
        "suffix_beta" : "Jour2_band1330_auto",
    },
    {
        "id"          : "3",
        "label"       : "Jour 3 seul",
        "mode"        : "loro",
        "dossier_mu"  : "npz_all_subjects_day3",
        "dossier_beta": "npz_all_subjects_day3",
        "suffix_mu"   : "Jour3_band0812_auto",
        "suffix_beta" : "Jour3_band1330_auto",
    },
    # ── Configs multi-jours → mode Train/Test inter-sessions ──
    {
        "id"         : "1+2",
        "label"      : "Jour 1 + Jour 2",
        "mode"       : "traintest",  # entraîne sur J1, teste sur J2
        "train_mu"   : ("npz_all_subjects_day1", "Jour1_band0812_motor8_guessB"),
        "train_beta" : ("npz_all_subjects_day1", "Jour1_band1330_motor8_guessB"),
        "test_mu"    : ("npz_all_subjects_day2", "Jour2_band0812_auto"),
        "test_beta"  : ("npz_all_subjects_day2", "Jour2_band1330_auto"),
    },
    {
        "id"         : "1+3",
        "label"      : "Jour 1 + Jour 3",
        "mode"       : "traintest",  # entraîne sur J1, teste sur J3
        "train_mu"   : ("npz_all_subjects_day1", "Jour1_band0812_motor8_guessB"),
        "train_beta" : ("npz_all_subjects_day1", "Jour1_band1330_motor8_guessB"),
        "test_mu"    : ("npz_all_subjects_day3", "Jour3_band0812_auto"),
        "test_beta"  : ("npz_all_subjects_day3", "Jour3_band1330_auto"),
    },
    {
        "id"         : "2+3",
        "label"      : "Jour 2 + Jour 3",
        "mode"       : "traintest",  # entraîne sur J2, teste sur J3
        "train_mu"   : ("npz_all_subjects_day2", "Jour2_band0812_auto"),
        "train_beta" : ("npz_all_subjects_day2", "Jour2_band1330_auto"),
        "test_mu"    : ("npz_all_subjects_day3", "Jour3_band0812_auto"),
        "test_beta"  : ("npz_all_subjects_day3", "Jour3_band1330_auto"),
    },
    {
        "id"         : "1+2+3",
        "label"      : "Tous les jours",
        "mode"       : "traintest",  # entraîne sur J1+J2 fusionnés, teste sur J3
        "train_mu"   : ("npz_all_subjects_day12", "Jour12_band0812_merged"),
        "train_beta" : ("npz_all_subjects_day12", "Jour12_band1330_merged"),
        "test_mu"    : ("npz_all_subjects_day3",  "Jour3_band0812_auto"),
        "test_beta"  : ("npz_all_subjects_day3",  "Jour3_band1330_auto"),
    },
]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              FONCTIONS UTILITAIRES                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def find_npz(dossier, sujet, suffix):
    """
    Recherche le fichier .npz correspondant à un sujet et un suffixe.

    Stratégie de recherche :
        1. Essai direct avec le pattern exact : {sujet}_{suffix}.npz
        2. Si non trouvé, recherche insensible à la casse
           (utile car certains sujets ont une casse différente, ex: subj06)

    Pourquoi la recherche insensible à la casse :
        Dans les données de Bilal, certains sujets ont des noms avec des
        majuscules inconsistantes ("Subj05" vs "subj06"). Le glob insensible
        à la casse permet de retrouver les fichiers même si la casse ne
        correspond pas exactement.

    Args:
        dossier : sous-dossier dans NPZ_BASE (ex: "npz_all_subjects_day1").
        sujet   : identifiant du sujet (ex: "Subj05", "subj06").
        suffix  : suffixe du fichier (ex: "Jour1_band0812_motor8_guessB").

    Returns:
        str : chemin absolu vers le fichier .npz, ou None si non trouvé.
    """
    # construction du chemin complet vers le dossier .npz
    base = Path(NPZ_BASE) / dossier

    # tentative 1 : pattern exact {sujet}_{suffix}.npz
    path = base / f"{sujet}_{suffix}.npz"
    if path.exists():
        return str(path)  # trouvé — on retourne directement

    # tentative 2 : recherche insensible à la casse avec glob
    # on cherche tous les fichiers finissant par _{suffix}.npz
    # et dont le nom commence par le sujet (en minuscules pour comparaison)
    for f in base.glob(f"*_{suffix}.npz"):
        if f.stem.lower().startswith(sujet.lower()):
            return str(f)  # trouvé avec casse différente

    # rien trouvé → retourner None (sera géré comme MANQUANT dans le main)
    return None


def parse_mean_bal_acc(output):
    """
    Extrait la valeur mean_bal_acc depuis la sortie texte d'EEGNet.

    Pourquoi parser la sortie texte :
        EEGNet est lancé en sous-processus et communique ses résultats
        via stdout/stderr. On capture cette sortie et on cherche la ligne
        "mean_acc=X.XXXXXX | mean_bal_acc=X.XXXXXX" pour extraire le score.

    Pattern recherché dans la sortie :
        "==== GLOBAL SUMMARY ===="
        "mean_acc=0.649908 | mean_bal_acc=0.475347"
        → on extrait 0.475347

    Pourquoi mean_bal_acc et pas mean_acc :
        La balanced accuracy pénalise équitablement toutes les classes.
        Avec ~50% de trials IDLE, mean_acc serait gonflée artificiellement
        (un classifieur naïf IDLE obtient déjà ~50% d'accuracy brute).

    Args:
        output : sortie texte complète du script EEGNet (stdout + stderr).

    Returns:
        float : valeur mean_bal_acc, ou None si non trouvée.
    """
    for line in output.split('\n'):
        # on cherche la ligne contenant mean_bal_acc
        if 'mean_bal_acc' in line:
            # extraction avec regex : cherche un float après "mean_bal_acc="
            m = re.search(r'mean_bal_acc=([0-9.]+)', line)
            if m:
                return float(m.group(1))  # conversion string → float
    # si aucune ligne ne contient mean_bal_acc → problème dans EEGNet
    return None


def run_eegnet(cmd):
    """
    Lance le script EEGNet en sous-processus et retourne sa sortie.

    Pourquoi subprocess plutôt qu'import direct :
        Chaque appel à EEGNet réinitialise complètement le modèle, les
        seeds et la mémoire GPU. Un import direct garderait des états
        résiduels entre les sujets. Le sous-processus garantit une
        isolation totale entre chaque évaluation.

    Timeout :
        600 secondes (10 minutes) par sujet. EEGNet prend ~3-11 minutes
        selon le nombre de runs et l'aug_factor. Si le timeout est dépassé,
        le sujet est marqué ERREUR et on passe au suivant.

    Capture de la sortie :
        capture_output=True capture stdout et stderr séparément.
        On les fusionne car EEGNet peut écrire dans l'un ou l'autre
        selon les warnings PyTorch.

    Args:
        cmd : liste de strings représentant la commande à exécuter
              (sans l'exécutable Python — ajouté par sys.executable).

    Returns:
        str : sortie complète (stdout + stderr) ou message d'erreur.
    """
    try:
        result = subprocess.run(
            [sys.executable] + cmd,  # sys.executable = même Python que le script courant
            capture_output=True,     # capture stdout ET stderr
            text=True,               # décode en string (pas bytes)
            timeout=600              # timeout de 10 minutes max
        )
        # fusion stdout + stderr pour avoir toute la sortie
        return result.stdout + result.stderr

    except subprocess.TimeoutExpired:
        # EEGNet a pris plus de 10 minutes → on abandonne ce sujet
        return "TIMEOUT"

    except Exception as e:
        # autre erreur (fichier introuvable, erreur Python, etc.)
        return f"ERROR: {e}"


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                                    MAIN                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# affichage de l'en-tête
print(f"\n{'='*70}")
print("EEGNET — 8 SUJETS RETENUS")
print("Configs : 1, 2, 3, 1+2, 1+3, 2+3, 1+2+3")
print(f"{'='*70}")

# stockage de tous les résultats pour le CSV final
all_rows = []

# dictionnaire pour accumuler les performances par config
# (pour calculer la moyenne sur les 8 sujets à la fin)
config_means = {cfg["id"]: [] for cfg in CONFIGS}

# ── Boucle principale : config × sujet ──
for cfg in CONFIGS:
    # affichage de la config courante
    print(f"\n{'='*60}")
    print(f"CONFIG : {cfg['id']} — {cfg['label']}")
    print(f"{'='*60}")
    print(f"  {'Sujet':<12} {'Mean bal acc':>14} {'Statut':>10}")
    print(f"  {'-'*40}")

    # boucle sur chaque sujet retenu
    for sujet in SUJETS:

        # ── Construction de la commande selon le mode ──
        if cfg["mode"] == "loro":
            # Mode LORO : un seul fichier .npz MU + un seul BETA
            # EEGNet va lui-même faire la rotation LORO sur les runs

            # recherche des fichiers .npz pour ce sujet
            npz_mu   = find_npz(cfg["dossier_mu"],  sujet, cfg["suffix_mu"])
            npz_beta = find_npz(cfg["dossier_beta"], sujet, cfg["suffix_beta"])

            # si l'un des fichiers est manquant → skip ce sujet
            if not npz_mu or not npz_beta:
                print(f"  {sujet:<12} {'N/A':>14} {'MANQUANT':>10}")
                all_rows.append({
                    "sujet": sujet, "config": cfg["id"],
                    "config_label": cfg["label"],
                    "mean_bal_acc": "N/A", "statut": "MANQUANT"
                })
                continue  # passer au sujet suivant

            # construction de la commande LORO
            # on passe les hyperparamètres optimaux déterminés empiriquement
            cmd = [
                EEGNET,
                "--npz_mu",   npz_mu,    # bande MU pour le gate
                "--npz_beta", npz_beta,  # bande BETA pour l'expert
                "--idle_windows",        "nomove1,nomove2",  # fenêtres repos
                "--move_windows",        "move1,move4",      # fenêtres mouvement (gate)
                "--move_windows_expert", "move1",            # fenêtre mouvement (expert)
                "--stack_mode_gate",     "time",   # concaténation temporelle move1+move4
                "--stack_mode_expert",   "time",
                "--gate_train_zscore",             # normalisation z-score sur train gate
                "--expert_train_zscore",           # normalisation z-score sur train expert
                "--epochs",     "150",   # max 150 epochs (early stopping actif)
                "--patience",   "30",    # arrêt si pas d'amélioration pendant 30 epochs
                "--batch_size", "32",    # petits batchs adaptés au faible volume EEG
                "--expert_aug_factor", "5",  # ×6 les trials expert par augmentation
                "--seed", "42",          # seed fixe pour reproductibilité
            ]

        else:
            # Mode Train/Test : fichiers séparés pour train et test
            # Entraîne sur un ou plusieurs jours, teste sur un autre jour

            # recherche des 4 fichiers .npz nécessaires (train + test, MU + BETA)
            train_mu   = find_npz(cfg["train_mu"][0],  sujet, cfg["train_mu"][1])
            train_beta = find_npz(cfg["train_beta"][0], sujet, cfg["train_beta"][1])
            test_mu    = find_npz(cfg["test_mu"][0],   sujet, cfg["test_mu"][1])
            test_beta  = find_npz(cfg["test_beta"][0],  sujet, cfg["test_beta"][1])

            # si l'un des 4 fichiers est manquant → skip ce sujet
            if not all([train_mu, train_beta, test_mu, test_beta]):
                print(f"  {sujet:<12} {'N/A':>14} {'MANQUANT':>10}")
                all_rows.append({
                    "sujet": sujet, "config": cfg["id"],
                    "config_label": cfg["label"],
                    "mean_bal_acc": "N/A", "statut": "MANQUANT"
                })
                continue  # passer au sujet suivant

            # construction de la commande Train/Test
            cmd = [
                EEGNET,
                "--npz_mu_train",   train_mu,    # MU pour l'entraînement
                "--npz_beta_train", train_beta,  # BETA pour l'entraînement
                "--npz_mu_test",    test_mu,     # MU pour le test
                "--npz_beta_test",  test_beta,   # BETA pour le test
                "--idle_windows",        "nomove1,nomove2",
                "--move_windows",        "move1,move4",
                "--move_windows_expert", "move1",
                "--stack_mode_gate",     "time",
                "--stack_mode_expert",   "time",
                "--gate_train_zscore",
                "--expert_train_zscore",
                "--epochs",     "150",
                "--patience",   "30",
                "--batch_size", "32",
                "--expert_aug_factor", "5",
                "--seed", "42",
            ]

        # ── Lancement d'EEGNet en sous-processus ──
        output  = run_eegnet(cmd)              # appel subprocess → récupère la sortie
        bal_acc = parse_mean_bal_acc(output)   # extraction du score depuis la sortie texte

        if bal_acc is not None:
            # succès → affichage et accumulation du résultat
            print(f"  {sujet:<12} {bal_acc:>14.3f} {'✅':>10}")
            config_means[cfg["id"]].append(bal_acc)  # pour calculer la moyenne globale
            statut = "OK"
        else:
            # échec → affichage des 200 derniers caractères de la sortie pour diagnostiquer
            print(f"  {sujet:<12} {'ERREUR':>14} {'❌':>10}")
            print(f"    Output: {output[-200:]}")
            bal_acc = None
            statut  = "ERREUR"

        # sauvegarde du résultat dans la liste pour le CSV
        all_rows.append({
            "sujet"       : sujet,
            "config"      : cfg["id"],
            "config_label": cfg["label"],
            "mean_bal_acc": f"{bal_acc:.3f}" if bal_acc else "N/A",
            "statut"      : statut,
        })


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              RÉSUMÉ GLOBAL                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print(f"\n\n{'='*70}")
print("RÉSUMÉ GLOBAL — EEGNet 8 sujets")
print(f"{'='*70}\n")

# construction de la liste des ids de config pour l'affichage
cfg_ids = [cfg["id"] for cfg in CONFIGS]

# affichage de l'en-tête du tableau (noms de colonnes)
print(f"{'Sujet':<12}", end="")
for cfg_id in cfg_ids:
    print(f"{'Cfg '+cfg_id:>10}", end="")
print()
print("-" * (12 + 10 * len(cfg_ids)))

# affichage des résultats par sujet
for sujet in SUJETS:
    print(f"  {sujet:<10}", end="")
    for cfg_id in cfg_ids:
        # recherche du résultat pour ce sujet × cette config
        row = next((r for r in all_rows
                    if r["sujet"] == sujet and r["config"] == cfg_id), None)
        val = row["mean_bal_acc"] if row else "N/A"
        print(f"{val:>10}", end="")
    print()

# affichage de la moyenne sur tous les sujets par config
print(f"\n  {'MEAN':<10}", end="")
for cfg_id in cfg_ids:
    vals = config_means[cfg_id]
    if vals:
        import numpy as np
        # moyenne arrondie à 3 décimales
        print(f"{np.mean(vals):.3f}".rjust(10), end="")
    else:
        print(f"{'N/A':>10}", end="")
print()

# affichage des résultats RLDA pour comparaison directe
# (valeurs calculées avec rlda_excellents.py sur les mêmes 5 sujets excellents)
print(f"\n  {'RLDA':<10}", end="")
rlda_ref = {
    "1"    : "0.725",   # RLDA Jour 1 seul
    "2"    : "0.754",   # RLDA Jour 2 seul
    "3"    : "0.762",   # RLDA Jour 3 seul
    "1+2"  : "0.775",   # RLDA Jour 1+2
    "1+3"  : "0.765",   # RLDA Jour 1+3
    "2+3"  : "0.773",   # RLDA Jour 2+3
    "1+2+3": "0.783",   # RLDA Tous les jours
}
for cfg_id in cfg_ids:
    print(f"{rlda_ref.get(cfg_id, '—'):>10}", end="")
print()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           SAUVEGARDE CSV                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

if all_rows:
    # ouverture du fichier CSV en écriture
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        # création du writer avec les colonnes dans l'ordre du premier dict
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()    # écriture de la ligne d'en-tête
        writer.writerows(all_rows)  # écriture de toutes les lignes de résultats
    print(f"\n✅ Résultats sauvegardés : {OUTPUT_CSV}")

print(f"{'='*70}\n")