#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              FUSION AUTOMATIQUE DES FICHIERS .npz MULTI-JOURS               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Objectif :                                                                  ║
║  Fusionner les fichiers .npz de deux jours différents pour créer les         ║
║  configurations multi-jours nécessaires à l'évaluation EEGNet.              ║
║                                                                              ║
║  Pourquoi fusionner les jours :                                              ║
║  Les configurations 1+2, 1+3, 2+3 et 1+2+3 d'Alchalabi et al. (2021)       ║
║  nécessitent de combiner les données de plusieurs sessions d'acquisition.    ║
║  Ce script crée automatiquement les fichiers .npz fusionnés pour tous        ║
║  les sujets disponibles.                                                     ║
║                                                                              ║
║  Exemple de fusion Jour1 + Jour2 :                                           ║
║    Subj05_Jour1_band0812_motor8_guessB.npz  ──┐                             ║
║                                               ├──→ Subj05_Jour12_band0812_merged.npz ║
║    Subj05_Jour2_band0812_auto.npz           ──┘                             ║
║                                                                              ║
║  Deux bandes fusionnées séparément :                                         ║
║    - band0812 (MU 8-12Hz)  → fichier *_band0812_merged.npz                  ║
║    - band1330 (BETA 13-30Hz) → fichier *_band1330_merged.npz                ║
║                                                                              ║
║  Usage :                                                                     ║
║    python merge_all_subjects_days.py                                         ║
║        --day1_dir npz_all_subjects_day1/                                     ║
║        --day2_dir npz_all_subjects_day2/                                     ║
║        --out_dir  npz_all_subjects_day12/                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import argparse  # pour parser les arguments en ligne de commande
import re        # pour extraire sujet et bande depuis le nom de fichier
from pathlib import Path  # pour manipuler les chemins de façon robuste

# import de la fonction de fusion depuis le module utils
# merge_npz prend deux fichiers .npz et les concatène en un seul
from utils.merge_npz_days import merge_npz


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                        PARSING DES NOMS DE FICHIERS                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def parse_npz_name(filename: str):
    """
    Extrait le nom du sujet et la bande fréquentielle depuis le nom d'un fichier .npz.

    Pourquoi parser le nom plutôt que hardcoder :
        Les fichiers .npz sont nommés selon une convention qui encode le sujet
        et la bande dans le nom. En parsant automatiquement, on n'a pas besoin
        de maintenir une liste manuelle des sujets — on découvre tout depuis
        les fichiers présents dans le dossier.

    Format attendu du nom de fichier :
        {Sujet}_Jour{N}_band{XXXX}_{suffixe}.npz

    Exemples concrets :
        Subj04_Jour1_band0812_motor8_guessB.npz → ("Subj04", "0812")
        Subj05_Jour2_band1330_auto.npz          → ("Subj05", "1330")
        subj06_Jour1_band0812_motor8_guessB.npz → ("subj06", "0812")

    Note sur la regex :
        (.+?)  : capture le nom du sujet (non-greedy pour s'arrêter au premier _Jour)
        (\d+)  : capture les 4 chiffres de la bande (0812 ou 1330)
        .*     : ignore le reste du nom (suffixe variable selon le jour)

    Args:
        filename : nom du fichier .npz (sans le chemin, juste le nom).

    Returns:
        tuple (sujet, bande) si le format est reconnu, None sinon.
        Ex: ("Subj04", "0812") ou None si format invalide.
    """
    # regex qui capture le sujet et la bande depuis le nom standardisé
    m = re.match(r"^(.+?)_Jour\d+_band(\d+)_.*\.npz$", filename)

    # si le nom ne correspond pas au pattern → fichier ignoré
    if not m:
        return None

    subject = m.group(1)  # groupe 1 = nom du sujet (ex: "Subj04")
    band    = m.group(2)  # groupe 2 = bande fréquentielle (ex: "0812")

    return subject, band


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                       CONSTRUCTION DE L'INDEX DES FICHIERS                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def build_index(folder: Path):
    """
    Construit un dictionnaire d'accès rapide aux fichiers .npz d'un dossier.

    Pourquoi un index plutôt qu'une recherche directe :
        Pour chaque sujet, on a besoin d'accéder rapidement à ses fichiers
        MU (band0812) et BETA (band1330). Construire un index une seule fois
        au départ est plus efficace que de refaire un glob() pour chaque sujet.

    Structure de l'index retourné :
        {
            ("Subj04", "0812") : Path("npz_day1/Subj04_Jour1_band0812_...npz"),
            ("Subj04", "1330") : Path("npz_day1/Subj04_Jour1_band1330_...npz"),
            ("Subj05", "0812") : Path("npz_day1/Subj05_Jour1_band0812_...npz"),
            ...
        }

    Utilisation de l'index :
        mu_file = index[("Subj04", "0812")]   → accès direct en O(1)

    Args:
        folder : Path vers le dossier contenant les fichiers .npz.

    Returns:
        dict : mapping (sujet, bande) → Path du fichier .npz.
    """
    index = {}  # dictionnaire vide qui sera rempli au fur et à mesure

    # on itère sur tous les fichiers .npz du dossier, triés alphabétiquement
    for f in sorted(folder.glob("*.npz")):

        # on essaie de parser le nom du fichier
        parsed = parse_npz_name(f.name)

        # si le nom ne correspond pas au format attendu → on passe au suivant
        if parsed is None:
            continue

        # on enregistre ce fichier dans l'index avec (sujet, bande) comme clé
        index[parsed] = f

    return index


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              PROGRAMME PRINCIPAL                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main():
    """
    Orchestre la fusion complète des fichiers .npz pour tous les sujets.

    Pipeline complet :
        1. Lecture des arguments CLI (chemins des dossiers)
        2. Vérification que les dossiers existent
        3. Construction des index pour Jour1 et Jour2
        4. Pour chaque sujet ayant les données des deux jours :
            a. Fusion du fichier MU  (band0812) → *_band0812_merged.npz
            b. Fusion du fichier BETA (band1330) → *_band1330_merged.npz
        5. Résumé final (OK / skipped / failed)

    Gestion des données manquantes :
        Si un sujet n'a pas les fichiers des deux jours (ex: subj06 sans Jour 3),
        il est simplement ignoré (SKIP) sans planter le script.
    """

    # ── Définition des arguments en ligne de commande ──
    ap = argparse.ArgumentParser(
        description="Fusionne automatiquement les NPZ de deux jours pour tous les sujets."
    )
    ap.add_argument("--day1_dir", required=True, type=str,
                    help="Dossier contenant les .npz du premier jour")
    ap.add_argument("--day2_dir", required=True, type=str,
                    help="Dossier contenant les .npz du second jour")
    ap.add_argument("--out_dir",  required=True, type=str,
                    help="Dossier de sortie pour les fichiers fusionnés")
    args = ap.parse_args()

    # ── Conversion des chemins en objets Path ──
    # Path est plus robuste que les strings pour manipuler les chemins
    # (gère automatiquement les séparateurs / et \ selon l'OS)
    day1_dir = Path(args.day1_dir)
    day2_dir = Path(args.day2_dir)
    out_dir  = Path(args.out_dir)

    # ── Vérification de l'existence des dossiers source ──
    if not day1_dir.exists():
        raise FileNotFoundError(f"Dossier Jour1 introuvable : {day1_dir}")
    if not day2_dir.exists():
        raise FileNotFoundError(f"Dossier Jour2 introuvable : {day2_dir}")

    # création du dossier de sortie s'il n'existe pas encore
    # parents=True crée aussi les dossiers parents si nécessaire
    # exist_ok=True ne plante pas si le dossier existe déjà
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Construction des index de fichiers ──
    # idx1 : tous les fichiers du Jour1, indexés par (sujet, bande)
    # idx2 : tous les fichiers du Jour2, indexés par (sujet, bande)
    idx1 = build_index(day1_dir)
    idx2 = build_index(day2_dir)

    # on récupère la liste des sujets présents dans le Jour1
    # (ce sont les sujets de référence — on fusionne pour ceux qui ont les deux jours)
    subjects = sorted(set(subj for subj, _ in idx1.keys()))

    # compteurs pour le résumé final
    n_ok   = 0  # nombre de sujets fusionnés avec succès
    n_skip = 0  # nombre de sujets ignorés (données manquantes)
    n_fail = 0  # nombre de sujets en erreur (problème technique)

    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║                   BOUCLE SUR TOUS LES SUJETS                        ║
    # ╚══════════════════════════════════════════════════════════════════════╝
    for subj in subjects:

        # récupération des 4 fichiers nécessaires pour ce sujet :
        # MU  Jour1, BETA Jour1, MU  Jour2, BETA Jour2
        mu1   = idx1.get((subj, "0812"))  # bande MU   du Jour1
        beta1 = idx1.get((subj, "1330"))  # bande BETA du Jour1
        mu2   = idx2.get((subj, "0812"))  # bande MU   du Jour2
        beta2 = idx2.get((subj, "1330"))  # bande BETA du Jour2

        # ── Vérification des fichiers du Jour1 ──
        # si MU ou BETA du Jour1 manquent → impossible de fusionner
        if mu1 is None or beta1 is None:
            print(f"[SKIP] {subj} — fichiers Jour1 manquants (mu={mu1}, beta={beta1})")
            n_skip += 1
            continue  # on passe au sujet suivant

        # ── Vérification des fichiers du Jour2 ──
        # certains sujets n'ont pas de Jour2 (ex: subj06 sans Jour3)
        if mu2 is None or beta2 is None:
            print(f"[SKIP] {subj} — fichiers Jour2 manquants (mu={mu2}, beta={beta2})")
            n_skip += 1
            continue  # on passe au sujet suivant

        # ── Définition des fichiers de sortie ──
        # convention de nommage : {sujet}_Jour12_band{XXXX}_merged.npz
        out_mu   = out_dir / f"{subj}_Jour12_band0812_merged.npz"
        out_beta = out_dir / f"{subj}_Jour12_band1330_merged.npz"

        # affichage du récapitulatif pour ce sujet avant fusion
        print("\n" + "=" * 70)
        print(f"[MERGE] {subj}")
        print(f"  Jour1 MU   : {mu1.name}")
        print(f"  Jour2 MU   : {mu2.name}")
        print(f"  Sortie MU  : {out_mu.name}")
        print(f"  Jour1 BETA : {beta1.name}")
        print(f"  Jour2 BETA : {beta2.name}")
        print(f"  Sortie BETA: {out_beta.name}")

        try:
            # ── Fusion de la bande MU (8-12Hz) ──
            # merge_npz concatène les runs des deux jours en un seul fichier
            merge_npz(mu1, mu2, out_mu)

            # ── Fusion de la bande BETA (13-30Hz) ──
            # même opération pour la bande beta
            merge_npz(beta1, beta2, out_beta)

            print(f"[OK] {subj} — fusion réussie")
            n_ok += 1  # on incrémente le compteur de succès

        except Exception as e:
            # capture de toute erreur inattendue (fichier corrompu, mémoire, etc.)
            # on affiche l'erreur mais on continue avec les autres sujets
            print(f"[FAIL] {subj} → {e}")
            n_fail += 1  # on incrémente le compteur d'échecs

    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║                          RÉSUMÉ FINAL                               ║
    # ╚══════════════════════════════════════════════════════════════════════╝
    print("\n" + "=" * 70)
    print("[RÉSUMÉ FINAL]")
    print(f"  Sujets fusionnés avec succès : {n_ok}")
    print(f"  Sujets ignorés (manquants)   : {n_skip}")
    print(f"  Sujets en erreur             : {n_fail}")
    print(f"  Total traités                : {n_ok + n_skip + n_fail}")
    print("=" * 70)


# ── Point d'entrée du script ──
# Ce bloc garantit que main() n'est appelée que si le script est exécuté
# directement (pas si importé comme module dans un autre script)
if __name__ == "__main__":
    main()