#!/usr/bin/env python3
"""
visualisation_eeg_live.py — Monitoring EEG en temps réel (Casque Kaptics)
=========================================================================
Développé dans le cadre du projet BCI-EEG ÉTS (2026)
Auteure : Emma

OBJECTIF :
    Script de monitoring EN TEMPS RÉEL à utiliser PENDANT l'installation
    du casque Kaptics sur le participant. Le graphique se met à jour
    automatiquement toutes les 2 secondes pour aider l'expérimentateur
    à ajuster le positionnement des électrodes jusqu'à obtenir un signal
    de qualité suffisante.

    Contrairement à analyse_kaptics.py qui analyse un fichier déjà
    enregistré, ce script surveille en continu le dernier fichier CSV
    produit par le logiciel Kaptics et affiche les métriques en direct.

WORKFLOW :
    1. Démarrer l'enregistrement dans le logiciel Kaptics
    2. Lancer ce script dans un terminal
    3. Observer les barres STD et ajuster le casque
    4. Quand le score est > 70 et C3/CZ/C4 sont verts → appuyer sur ENTRÉE
    5. Lancer l'acquisition du protocole expérimental

INTERPRÉTATION DU SCORE GLOBAL (/100) :
    > 70  → ✅ GO — Signal valide, lancer l'acquisition
    40-70 → ⚠️  LIMITE — Ajuster le casque
    < 40  → ❌ MAUVAIS — Repositionner, attendre la stabilisation

CODE COULEUR DES BARRES :
    🟢 Vert   (std ≥ 15 µV)  : bon contact
    🟡 Orange (std 5-15 µV)  : contact acceptable
    ⬛ Gris   (std < 5 µV)   : mauvais contact
    🔴 Rouge  (std > 150 µV) : artefact
    🟣 Violet (std > 500 µV) : saturation électronique

COMMANDES :
    [ENTRÉE] → Valider le signal et arrêter le monitoring
    [q]      → Quitter sans valider

IMPORTANT — CHEMIN DES DONNÉES :
    Modifier la variable FOLDER ci-dessous pour pointer vers le dossier
    où le logiciel Kaptics enregistre les fichiers CSV sur votre machine.

DÉPENDANCES :
    pip install numpy pandas matplotlib scipy

NOTE OS :
    Le chemin par défaut est configuré pour Windows (C:\\Kaptics\\records).
    Sur Mac/Linux, utiliser un chemin du type : /Users/nom/Kaptics/records
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import glob
import time
import threading

# ======================
# CONFIG
# ======================

# ⚠️  MODIFIER CE CHEMIN selon votre système d'exploitation et installation :
# Windows : r"C:\Kaptics\records"
# Mac     : "/Users/votre_nom/Kaptics/records"
# Linux   : "/home/votre_nom/Kaptics/records"
FOLDER = r"C:\Kaptics\records"

# Fenêtre d'analyse : nombre d'échantillons analysés à chaque rafraîchissement
# 1250 échantillons à 250 Hz = 5 secondes de signal
# Augmenter pour plus de stabilité, diminuer pour plus de réactivité
WINDOW = 1250

# Intervalle de rafraîchissement du graphique en secondes
# 2.0s = bon compromis entre réactivité et lisibilité
REFRESH = 2.0

# Mapping entre les noms courts des canaux et les noms complets dans le CSV
# Les canaux C3, CZ, C4 sont les plus importants pour l'imagerie motrice
CHANNEL_MAP = {
    "C3": "Channel7 (C3)",
    "CZ": "Channel9 (CZ)",
    "C4": "Channel11 (C4)"
}

# ======================
# ÉTAT PARTAGÉ
# ======================

# Dictionnaire partagé entre les threads monitoring et input
# Permet au thread de monitoring de communiquer l'état au thread d'input
state = {
    "running": True,       # True tant que le monitoring est actif
    "validated": False,    # True quand l'utilisateur valide le signal
    "current_file": None,  # Chemin du fichier CSV analysé
    "last_std": None,      # Derniers STD calculés (array numpy)
    "last_score": 0,       # Dernier score global /100
    "last_status": "",     # Dernier statut textuel
}

# ======================
# DERNIER FICHIER
# ======================

def get_latest_file():
    """
    Trouve le fichier CSV le plus récent dans le dossier FOLDER.
    Le logiciel Kaptics crée un nouveau fichier à chaque enregistrement.
    On surveille toujours le fichier le plus récent (trié par date de modification).
    """
    files = glob.glob(os.path.join(FOLDER, "*.csv"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)

# ======================
# ANALYSE DU SIGNAL
# ======================

def analyze(df, eeg_cols):
    """
    Analyse les WINDOW derniers échantillons du signal EEG.

    Calcule pour chaque canal :
    - std (µV) : écart-type = indicateur principal de qualité du contact
    - Catégorie : BON / MOYEN / FAIBLE / ARTEFACT / SATURATION

    Calcule un score global /100 basé sur :
    - +2 points par canal BON (std 15-150 µV)
    - +1 point par canal MOYEN (std 5-15 µV)
    - -2 points par canal FAIBLE (std < 5 µV)
    - -3 points par canal avec artefact (std 150-500 µV)
    - -5 points par canal saturé (std > 500 µV)

    Le statut final dépend du score ET de l'état des canaux C3/CZ/C4 :
    - Si saturation → STOP immédiat
    - Si C3/CZ/C4 faibles → CANAUX MOTEURS KO même si score OK

    Args:
        df       : DataFrame avec les données EEG
        eeg_cols : liste des colonnes EEG

    Returns:
        dict avec std, score, status, c3, cz, c4 ou None si données insuffisantes
    """
    if len(df) < WINDOW:
        return None

    # Analyse sur les WINDOW derniers échantillons (5 secondes)
    data = df.tail(WINDOW)
    eeg = np.nan_to_num(data[eeg_cols].values)
    std = np.std(eeg, axis=0)

    # Classification de chaque canal
    good_mask    = (std >= 15)  & (std <= 150)   # signal physiologique normal
    medium_mask  = (std >= 5)   & (std < 15)     # signal acceptable
    bad_mask     = std < 5                         # mauvais contact
    artefact_mask= (std > 150)  & (std <= 500)   # artefact de mouvement
    sat_mask     = std > 500                       # saturation électronique

    good      = int(np.sum(good_mask))
    medium    = int(np.sum(medium_mask))
    bad       = int(np.sum(bad_mask))
    artefacts = int(np.sum(artefact_mask))
    saturation= int(np.sum(sat_mask))

    # Score global normalisé sur 100
    # Maximum théorique = 16 canaux × 2 points = 32 → divisé pour avoir /100
    score = good*2 + medium*1 - bad*2 - artefacts*3 - saturation*5
    score = max(0, min(100, int((score / 32) * 100)))

    def get_std(name):
        """Récupère le std d'un canal par son nom court (C3, CZ, C4)."""
        col = CHANNEL_MAP[name]
        return std[eeg_cols.index(col)] if col in eeg_cols else np.nan

    c3, cz, c4 = get_std("C3"), get_std("CZ"), get_std("C4")

    # Détermination du statut final
    if saturation > 0:
        status = "STOP — SATURATION"
        status_color = "#7B0000"
    elif artefacts > 3:
        status = "MAUVAIS — ARTEFACTS"
        status_color = "#B71C1C"
    elif c3 < 5 or cz < 5 or c4 < 5:
        status = "CANAUX MOTEURS KO"
        status_color = "#E65100"
    elif score > 70:
        status = "GO — SIGNAL VALIDE"
        status_color = "#1B5E20"
    else:
        status = "LIMITE — AJUSTER"
        status_color = "#F57F17"

    return {
        "std": std,
        "eeg_cols": eeg_cols,
        "good": good, "medium": medium, "bad": bad,
        "artefacts": artefacts, "saturation": saturation,
        "score": score, "status": status, "status_color": status_color,
        "c3": c3, "cz": cz, "c4": c4,
    }

# ======================
# THREAD MONITORING
# ======================

def monitoring_loop(ax_bar, ax_motor, fig):
    """
    Boucle principale du monitoring — s'exécute dans un thread séparé.

    Toutes les REFRESH secondes :
    1. Charge le dernier fichier CSV Kaptics
    2. Analyse les derniers WINDOW échantillons
    3. Met à jour le graphique matplotlib

    Le thread s'arrête quand state["running"] = False
    (signal validé par l'utilisateur ou commande "q").
    """
    while state["running"] and not state["validated"]:
        try:
            new_file = get_latest_file()
            if new_file is None:
                time.sleep(REFRESH)
                continue

            if new_file != state["current_file"]:
                state["current_file"] = new_file

            df = pd.read_csv(state["current_file"], sep="\t")
            eeg_cols = [col for col in df.columns if "Channel" in col]

            result = analyze(df, eeg_cols)
            if result is None:
                time.sleep(REFRESH)
                continue

            # Mise à jour de l'état partagé (lu par le thread input)
            state["last_std"] = result["std"]
            state["last_score"] = result["score"]
            state["last_status"] = result["status"]

            std          = result["std"]
            score        = result["score"]
            status       = result["status"]
            status_color = result["status_color"]
            c3, cz, c4  = result["c3"], result["cz"], result["c4"]

            # Couleurs des barres selon la qualité du canal
            colors = []
            for s in std:
                if s > 500:   colors.append("#6A0DAD")   # violet = saturation
                elif s > 150: colors.append("#B71C1C")   # rouge  = artefact
                elif s >= 15: colors.append("#2E7D32")   # vert   = bon
                elif s >= 5:  colors.append("#F57F17")   # orange = moyen
                else:         colors.append("#424242")   # gris   = faible

            # ── Graphique gauche : STD par canal ──
            ax_bar.clear()
            labels = [col.split("(")[-1].replace(")", "") for col in eeg_cols]
            bars = ax_bar.bar(range(len(std)), std, color=colors, width=0.6, zorder=3)

            # Bordure bleue sur les canaux moteurs C3/CZ/C4
            for name, col in CHANNEL_MAP.items():
                if col in eeg_cols:
                    idx = eeg_cols.index(col)
                    bars[idx].set_edgecolor("#1565C0")
                    bars[idx].set_linewidth(2.5)

            ax_bar.set_xticks(range(len(std)))
            ax_bar.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
            ax_bar.axhline(15,  linestyle="--", color="#2E7D32", linewidth=1.2, label="Seuil BON (15 µV)")
            ax_bar.axhline(5,   linestyle="--", color="#F57F17", linewidth=1.2, label="Seuil MOYEN (5 µV)")
            ax_bar.axhline(150, linestyle="--", color="#B71C1C", linewidth=1,   label="Seuil artefact (150 µV)")
            ax_bar.set_ylim(0, max(200, np.max(std) * 1.15))
            ax_bar.set_ylabel("STD (µV)", fontsize=10)
            ax_bar.set_title(
                f"SCORE GLOBAL : {score}/100   —   {status}",
                fontsize=13, fontweight="bold", color=status_color, pad=10
            )
            ax_bar.grid(axis="y", alpha=0.3, zorder=0)

            legend_patches = [
                mpatches.Patch(color="#2E7D32", label="BON ≥ 15 µV"),
                mpatches.Patch(color="#F57F17", label="MOYEN 5–15 µV"),
                mpatches.Patch(color="#424242", label="FAIBLE < 5 µV"),
                mpatches.Patch(color="#B71C1C", label="ARTEFACT > 150 µV"),
                mpatches.Patch(color="#6A0DAD", label="SATURATION > 500 µV"),
            ]
            ax_bar.legend(handles=legend_patches, loc="upper right", fontsize=8)

            # ── Graphique droit : jauges C3/CZ/C4 ──
            ax_motor.clear()

            motor_vals   = [c3, cz, c4]
            motor_names  = ["C3", "CZ", "C4"]
            motor_colors = []
            for v in motor_vals:
                if np.isnan(v):    motor_colors.append("#9E9E9E")
                elif v >= 15:      motor_colors.append("#2E7D32")
                elif v >= 5:       motor_colors.append("#F57F17")
                else:              motor_colors.append("#B71C1C")

            bars_m = ax_motor.barh(motor_names, motor_vals, color=motor_colors, height=0.5, zorder=3)
            ax_motor.axvline(15, linestyle="--", color="#2E7D32", linewidth=1.2)
            ax_motor.axvline(5,  linestyle="--", color="#F57F17", linewidth=1.2)
            ax_motor.set_xlim(0, max(50, np.nanmax(motor_vals) * 1.2))
            ax_motor.set_xlabel("STD (µV)", fontsize=10)
            ax_motor.set_title("Canaux moteurs C3 / CZ / C4", fontsize=11, fontweight="bold")
            ax_motor.grid(axis="x", alpha=0.3, zorder=0)

            for i, (v, bar) in enumerate(zip(motor_vals, bars_m)):
                if not np.isnan(v):
                    label = f"{v:.1f} µV"
                    if v >= 15:   label += "  ✓ BON"
                    elif v >= 5:  label += "  ~ MOYEN"
                    else:         label += "  ✗ FAIBLE"
                    ax_motor.text(v + 0.5, bar.get_y() + bar.get_height()/2,
                                  label, va="center", fontsize=10, fontweight="bold")

            fig.suptitle(
                f"🟢 Bons: {result['good']}   🟡 Moyens: {result['medium']}   "
                f"🔴 Faibles: {result['bad']}   ⚠️ Artefacts: {result['artefacts']}   "
                f"🚨 Saturation: {result['saturation']}   "
                f"│   Fichier : {os.path.basename(state['current_file'])}",
                fontsize=9, color="#333333"
            )

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

        except Exception as e:
            print(f"[Erreur monitoring] {e}")

        time.sleep(REFRESH)

# ======================
# THREAD INPUT
# ======================

def input_loop():
    """
    Gère les commandes clavier de l'expérimentateur.

    S'exécute dans le thread principal (matplotlib doit être
    dans le thread principal sur certains OS).

    ENTRÉE → valide le signal et arrête le monitoring
    q      → quitte sans valider
    """
    print("\n" + "="*50)
    print("  🧠  MONITORING EEG EN TEMPS RÉEL")
    print("="*50)
    print(f"  Dossier surveillé : {FOLDER}")
    print(f"  Fenêtre d'analyse : {WINDOW} échantillons ({WINDOW/250:.0f}s à 250Hz)")
    print(f"  Rafraîchissement  : toutes les {REFRESH}s")
    print("="*50)
    print("\n  Le graphique se met à jour automatiquement.")
    print("  Ajustez le casque et regardez les STD monter.")
    print()
    print("  Commandes disponibles :")
    print("  [ENTRÉE]  → Valider le signal et arrêter le monitoring")
    print("  [q]       → Quitter sans valider")
    print()

    while state["running"]:
        cmd = input()
        if cmd.lower() == "q":
            print("\n❌ Session annulée.")
            state["running"] = False
            break
        else:
            score  = state["last_score"]
            status = state["last_status"]
            std    = state["last_std"]

            print("\n" + "="*50)
            print("  ✅  SIGNAL VALIDÉ — DÉMARRAGE ACQUISITION")
            print("="*50)
            print(f"  Score final   : {score}/100")
            print(f"  Status final  : {status}")

            if std is not None:
                try:
                    df = pd.read_csv(state["current_file"], sep="\t")
                    eeg_cols = [col for col in df.columns if "Channel" in col]
                    data = df.tail(WINDOW)
                    eeg = np.nan_to_num(data[eeg_cols].values)
                    std_final = np.std(eeg, axis=0)
                    print("\n  Détail canaux au moment de la validation :")
                    for i, s in enumerate(std_final):
                        name = eeg_cols[i]
                        short = name.split("(")[-1].replace(")", "")
                        if s > 500:    tag = "🚨 SATURATION"
                        elif s > 150:  tag = "⚠️  ARTEFACT"
                        elif s >= 15:  tag = "🟢 BON"
                        elif s >= 5:   tag = "🟡 MOYEN"
                        else:          tag = "🔴 FAIBLE"
                        print(f"    {short:<6} {s:>7.1f} µV   {tag}")
                except Exception as e:
                    print(f"  [Erreur détail final] {e}")

            print("\n  Vous pouvez maintenant lancer l'acquisition.\n")
            state["validated"] = True
            state["running"] = False
            break

# ======================
# MAIN
# ======================

if __name__ == "__main__":

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax_bar, ax_motor) = plt.subplots(
        1, 2,
        figsize=(16, 6),
        gridspec_kw={"width_ratios": [3, 1]}
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show(block=False)

    # Thread de monitoring (lecture CSV + mise à jour graphique)
    t_monitor = threading.Thread(
        target=monitoring_loop,
        args=(ax_bar, ax_motor, fig),
        daemon=True  # s'arrête automatiquement quand le programme principal se ferme
    )
    t_monitor.start()

    # Thread d'input dans le thread principal (requis par matplotlib)
    input_loop()

    plt.close("all")
