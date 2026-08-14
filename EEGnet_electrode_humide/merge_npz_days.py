#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    FUSION DE DEUX FICHIERS .npz EEG (JOURS)                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Objectif :                                                                  ║
║  Fusionner intelligemment deux fichiers .npz EEG (Jour1 + Jour2) en un      ║
║  seul fichier contenant les runs des deux jours concaténés.                  ║
║                                                                              ║
║  Pourquoi c'est complexe :                                                   ║
║  Les fichiers .npz EEG peuvent contenir des structures variées :             ║
║    - des runs sous forme de liste Python                                     ║
║    - des runs sous forme de numpy array d'objets                             ║
║    - des arrays numériques (features, labels)                                ║
║    - des scalaires (fréquence d'échantillonnage, bande)                      ║
║    - des métadonnées (meta)                                                  ║
║  Chaque cas nécessite une stratégie de fusion différente.                    ║
║                                                                              ║
║  Utilisé par : merge_all_subjects_days.py                                    ║
║                                                                              ║
║  Usage CLI direct (pour un seul fichier) :                                   ║
║    python merge_npz_days.py                                                  ║
║        --npz_day1 Subj05_Jour1_band0812_motor8_guessB.npz                   ║
║        --npz_day2 Subj05_Jour2_band0812_auto.npz                             ║
║        --out_npz  Subj05_Jour12_band0812_merged.npz                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import argparse        # pour le CLI (usage direct du script)
import numpy as np     # pour la manipulation des arrays et fichiers .npz
from pathlib import Path  # pour la gestion robuste des chemins


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              UTILITAIRES                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def load_npz_as_dict(path: Path):
    """
    Charge un fichier .npz et retourne son contenu sous forme de dict Python.

    Pourquoi convertir en dict :
        np.load() retourne un NpzFile — un objet lazy qui charge les arrays
        à la demande. En convertissant en dict, on charge tout en mémoire
        d'un coup ce qui est plus sûr pour la fusion (pas de fichier ouvert
        en arrière-plan pendant les opérations).

    Structure typique d'un fichier .npz EEG :
        {
            "runs"  : array d'objets (liste de runs EEG),
            "meta"  : métadonnées de la session,
            "fs"    : fréquence d'échantillonnage (256 Hz),
            "band"  : bande fréquentielle [lo, hi]
        }

    Args:
        path : Path vers le fichier .npz à charger.

    Returns:
        dict : clé → valeur numpy pour chaque champ du .npz.
    """
    obj = np.load(path, allow_pickle=True)  # allow_pickle=True pour les arrays d'objets
    # on construit un dict en copiant chaque valeur du NpzFile
    return {k: obj[k] for k in obj.files}


def is_runs_like_array(x):
    """
    Détermine si un array numpy contient des runs EEG (array d'objets).

    Pourquoi ce test :
        Les runs EEG sont stockés comme numpy array de dtype=object
        (car chaque run est un dict Python avec des clés variables).
        On les distingue des arrays numériques classiques par leur dtype.

    Exemples :
        np.array([run1, run2], dtype=object) → True  (runs EEG)
        np.array([1.0, 2.0], dtype=float32) → False  (array numérique)
        np.array(256, dtype=int)            → False  (scalaire)

    Args:
        x : valeur à tester.

    Returns:
        bool : True si c'est un array d'objets numpy (runs EEG).
    """
    return isinstance(x, np.ndarray) and x.dtype == object


def to_list_if_scalar_object(x):
    """
    Convertit un array numpy 0-dimensionnel en son contenu Python natif.

    Cas critique — array 0D :
        Quand numpy sauvegarde une liste Python dans un .npz, il peut
        la encapsuler dans un array de shape () et dtype=object.
        Exemple :
            np.array([run1, run2], dtype=object)   → sauvegardé comme array 0D
            après np.load → array(shape=(), dtype=object)
            après .item() → [run1, run2]  (la liste originale)

        Sans ce dépliage, la concaténation échoue car np.concatenate
        ne sait pas gérer les arrays 0D.

    Args:
        x : valeur potentiellement encapsulée dans un array 0D.

    Returns:
        Contenu Python natif si array 0D, sinon retourne x inchangé.
    """
    # un array de shape () est un array 0-dimensionnel → on déplie avec .item()
    if isinstance(x, np.ndarray) and x.shape == ():
        return x.item()  # extrait le contenu Python (liste, dict, etc.)
    return x  # pas un array 0D → on retourne tel quel


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           LOGIQUE DE FUSION                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def merge_values(v1, v2, key):
    """
    Fusionne deux valeurs d'un même champ selon leur type.

    Stratégie de fusion par cas :
        Les fichiers .npz EEG contiennent des champs de types très variés.
        Cette fonction implémente une stratégie adaptée à chaque type
        pour produire une fusion correcte.

    Cas traités (par ordre de priorité) :

        CAS SPÉCIAL — "meta" :
            Les métadonnées ne sont pas fusionnées — on garde celles du Jour1.
            Raison : les métadonnées sont spécifiques à une session et ne
            peuvent pas être meaningfully combinées.

        CAS 1 — Listes / tuples (runs Python) :
            Concaténation directe des deux listes.
            [run1, run2] + [run3, run4] → [run1, run2, run3, run4]

        CAS 2 — Arrays numpy d'objets (runs stockés en numpy) :
            np.concatenate si possible, sinon fallback sur list + array.
            Résultat : array([run1, run2, run3, run4], dtype=object)

        CAS 3 — Arrays numériques (features, labels) :
            Si identiques → on garde le premier (ex: fréquences spectrales).
            Sinon → np.concatenate sur l'axe 0 (ajout de trials).

        CAS 4 — Scalaires (fs, bande, etc.) :
            Si égaux → on garde le premier (ex: fs=256 dans les deux jours).
            Si différents → erreur car incompatible (ne devrait pas arriver).

        CAS 5 — Fallback arrays identiques :
            Si les deux arrays sont identiques → on garde le premier.
            Sinon → erreur.

    Args:
        v1  : valeur du champ dans le fichier Jour1.
        v2  : valeur du champ dans le fichier Jour2.
        key : nom du champ (utilisé pour les messages d'erreur).

    Returns:
        Valeur fusionnée selon la stratégie appropriée.

    Raises:
        ValueError : si les valeurs sont incompatibles (scalaires différents
                     ou type non géré).
    """

    # ── CAS SPÉCIAL : métadonnées ──
    # on ne fusionne jamais les métadonnées — elles sont propres à chaque session
    if key == "meta":
        return v1  # on garde les métadonnées du Jour1 par convention

    # ── Dépliage des arrays 0D si nécessaire ──
    # (cas où numpy a encapsulé une liste dans un array 0D lors de la sauvegarde)
    v1 = to_list_if_scalar_object(v1)
    v2 = to_list_if_scalar_object(v2)

    # ── CAS 1 : runs sous forme de liste ou tuple Python ──
    # on concatène simplement les deux listes et on retourne un array d'objets
    if isinstance(v1, (list, tuple)) and isinstance(v2, (list, tuple)):
        # list(v1) + list(v2) = concaténation des deux listes de runs
        # np.array(..., dtype=object) = conversion en array numpy pour homogénéité
        return np.array(list(v1) + list(v2), dtype=object)

    # ── CAS 2 : runs sous forme d'array numpy d'objets ──
    if is_runs_like_array(v1) and is_runs_like_array(v2):
        try:
            # tentative de concaténation directe numpy (plus efficace)
            return np.concatenate([v1, v2], axis=0)
        except Exception:
            # fallback si np.concatenate échoue (arrays de shapes incompatibles)
            # on passe par une liste Python intermédiaire
            return np.array(list(v1) + list(v2), dtype=object)

    # ── CAS 3 : arrays numériques (features PSD, labels, etc.) ──
    if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
        # sous-cas 3a : arrays identiques → on garde le premier
        # ex: vecteur de fréquences Welch identique dans les deux jours
        if v1.shape == v2.shape and np.array_equal(v1, v2):
            return v1  # pas besoin de dupliquer si identiques

        # sous-cas 3b : arrays différents → concaténation sur l'axe 0
        # ex: matrices de features (n_trials × n_features) → on empile les trials
        try:
            return np.concatenate([v1, v2], axis=0)
        except Exception:
            pass  # si la concaténation échoue → on tombe sur le cas 5

    # ── CAS 4 : scalaires (fréquence, bande fréquentielle, etc.) ──
    if np.isscalar(v1) and np.isscalar(v2):
        if v1 == v2:
            return v1  # scalaires identiques (ex: fs=256) → on garde le premier
        # scalaires différents → erreur car ça ne devrait jamais arriver
        # (les deux fichiers doivent avoir la même fréquence et la même bande)
        raise ValueError(
            f"Champ scalaire incompatible pour '{key}' : {v1} ≠ {v2}\n"
            f"Vérifiez que les deux fichiers ont la même bande et fréquence."
        )

    # ── CAS 5 : fallback — arrays identiques non gérés par les cas précédents ──
    if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
        if np.array_equal(v1, v2):
            return v1  # identiques → on garde le premier

    # aucun cas ne correspond → erreur explicite avec le nom du champ
    raise ValueError(
        f"Impossible de fusionner le champ '{key}' — "
        f"types : {type(v1).__name__} et {type(v2).__name__}"
    )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                            FUSION PRINCIPALE                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def merge_npz(day1_path: Path, day2_path: Path, out_path: Path):
    """
    Fusionne deux fichiers .npz EEG (Jour1 + Jour2) en un seul fichier.

    Algorithme :
        1. Chargement des deux fichiers en dict Python
        2. Identification des champs communs et exclusifs
        3. Fusion intelligente des champs communs (via merge_values)
        4. Inclusion des champs exclusifs (présents dans un seul jour)
        5. Sauvegarde du résultat compressé en .npz

    Gestion des champs exclusifs :
        Si un champ existe dans Jour1 mais pas Jour2 (ou vice-versa),
        on l'inclut tel quel dans le fichier fusionné sans erreur.
        Cela garantit qu'aucune information n'est perdue.

    Format de sortie :
        Fichier .npz compressé contenant les runs des deux jours
        concaténés dans la clé "runs".

    Args:
        day1_path : Path vers le fichier .npz du Jour1.
        day2_path : Path vers le fichier .npz du Jour2.
        out_path  : Path vers le fichier .npz de sortie.
    """
    # ── Chargement des deux fichiers ──
    d1 = load_npz_as_dict(day1_path)  # dict du Jour1
    d2 = load_npz_as_dict(day2_path)  # dict du Jour2

    # ── Analyse des clés ──
    keys1 = set(d1.keys())  # ensemble des clés du Jour1
    keys2 = set(d2.keys())  # ensemble des clés du Jour2

    common_keys = sorted(keys1 & keys2)  # clés présentes dans les DEUX jours → à fusionner
    only_day1   = sorted(keys1 - keys2)  # clés uniquement dans Jour1 → à garder telles quelles
    only_day2   = sorted(keys2 - keys1)  # clés uniquement dans Jour2 → à garder telles quelles

    merged = {}  # dict qui contiendra le résultat de la fusion

    # ── Fusion des champs communs ──
    # pour chaque champ présent dans les deux jours, on appelle merge_values
    # qui choisit la stratégie de fusion adaptée au type du champ
    for key in common_keys:
        merged[key] = merge_values(d1[key], d2[key], key)

    # ── Inclusion des champs exclusifs du Jour1 ──
    # ces champs n'existent que dans le Jour1 → on les garde tels quels
    for key in only_day1:
        merged[key] = d1[key]

    # ── Inclusion des champs exclusifs du Jour2 ──
    # ces champs n'existent que dans le Jour2 → on les ajoute au résultat
    for key in only_day2:
        merged[key] = d2[key]

    # ── Sauvegarde du fichier fusionné ──
    # on crée les dossiers parents si nécessaire
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # np.savez_compressed : sauvegarde compressée (plus petite que savez normal)
    # **merged décompresse le dict en arguments nommés
    np.savez_compressed(out_path, **merged)

    print(f"[OK] Fichier fusionné sauvegardé : {out_path}")

    # affichage informatif des champs non communs (pour débogage)
    if only_day1:
        print(f"     Champs gardés depuis Jour1 uniquement : {only_day1}")
    if only_day2:
        print(f"     Champs gardés depuis Jour2 uniquement : {only_day2}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                                   CLI                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main():
    """
    Point d'entrée CLI pour fusionner un seul fichier .npz manuellement.

    Usage typique :
        Pour fusionner manuellement un sujet spécifique sans passer
        par merge_all_subjects_days.py.

    Note :
        Pour fusionner tous les sujets d'un coup, utiliser
        merge_all_subjects_days.py qui appelle merge_npz en boucle.
    """
    ap = argparse.ArgumentParser(
        description="Fusionne deux fichiers .npz EEG (Jour1 + Jour2) en un seul."
    )
    # chemin vers le fichier du premier jour
    ap.add_argument("--npz_day1", required=True, type=str,
                    help="Chemin vers le .npz du Jour1")
    # chemin vers le fichier du second jour
    ap.add_argument("--npz_day2", required=True, type=str,
                    help="Chemin vers le .npz du Jour2")
    # chemin de sortie pour le fichier fusionné
    ap.add_argument("--out_npz",  required=True, type=str,
                    help="Chemin de sortie pour le .npz fusionné")
    args = ap.parse_args()

    # lancement de la fusion avec les chemins convertis en Path
    merge_npz(
        Path(args.npz_day1),  # conversion string → Path pour robustesse
        Path(args.npz_day2),
        Path(args.out_npz),
    )


# point d'entrée : s'exécute uniquement si lancé directement
# (pas si importé comme module par merge_all_subjects_days.py)
if __name__ == "__main__":
    main()