#!/usr/bin/env python3
"""
analyse_qualite_kaptics.py — Analyse qualité du signal EEG Kaptics
===================================================================
Développé dans le cadre du projet BCI-EEG ÉTS (2026)
Auteure : Emma

OBJECTIF :
    Ce script analyse la qualité du signal EEG enregistré par le casque
    à électrodes sèches Kaptics AVANT chaque session d'acquisition.
    Il permet de s'assurer que le contact électrode-peau est suffisant
    pour obtenir des données exploitables pour la classification BCI.

POURQUOI C'EST IMPORTANT :
    Contrairement aux électrodes humides (gel conducteur), les électrodes
    sèches du casque Kaptics n'ont pas de gel conducteur. La qualité du
    contact dépend directement de la pression mécanique et de la présence
    de cheveux entre l'électrode et le scalp. Un mauvais contact se traduit
    par un signal de faible amplitude (std < 5 µV) ou des artefacts
    (amplitude > 150 µV) qui rendront les données inexploitables.

    Les canaux les plus importants sont C3, CZ et C4, positionnés
    au-dessus du cortex moteur primaire — c'est là que les modulations
    ERD/ERS liées à l'imagerie motrice sont les plus discriminantes.

PRODUIT :
    1. Analyse quantitative (statistiques par canal dans le terminal)
    2. Graphique PNG avec 7 panneaux d'analyse
    3. Rapport texte avec verdict et recommandations

INTERPRÉTATION DES RÉSULTATS :
    - std > 15 µV  : ✅ BON contact — signal physiologique détectable
    - std 5-15 µV  : ⚠️  MOYEN — contact acceptable mais à améliorer
    - std < 5 µV   : ❌ FAIBLE — mauvais contact, repositionner l'électrode
    - amplitude > 150 µV : artefact électronique ou mouvement brusque

    Pour lancer une acquisition, les 3 canaux critiques (C3, CZ, C4)
    doivent avoir std ≥ 10 µV.

FORMAT DES DONNÉES :
    Le script attend un fichier CSV exporté par le logiciel Kaptics.
    Les colonnes EEG doivent contenir "Channel" dans leur nom.
    Les valeurs sont en nanovolts (nV) — converties en µV automatiquement.
    Une colonne "TimeStamp" est attendue pour calculer la fréquence
    d'échantillonnage. Une colonne "Marker" est utilisée pour détecter
    les marqueurs Unity si connecté.

USAGE :
    # Analyse d'un fichier CSV Kaptics
    python analyse_kaptics.py --file "/chemin/vers/session.csv"

    # Avec dossier de sortie personnalisé
    python analyse_kaptics.py --file "/chemin/vers/session.csv" --out "/chemin/vers/output/"

DÉPENDANCES :
    pip install numpy pandas matplotlib scipy

EXEMPLE DE WORKFLOW :
    1. Poser le casque Kaptics sur le participant
    2. Enregistrer 30 secondes de signal dans le logiciel Kaptics
    3. Exporter en CSV
    4. Lancer ce script pour vérifier la qualité
    5. Si verdict OK → lancer l'acquisition complète
    6. Si verdict INSUFFISANT → repositionner le casque et recommencer
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from scipy.signal import welch, butter, filtfilt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION — Modifier ces valeurs si nécessaire
# ============================================================

# Seuils de qualité du signal (en µV après conversion nV→µV)
# Ces valeurs ont été déterminées empiriquement lors des sessions
# d'acquisition Kaptics au laboratoire de l'ÉTS (2026)
SEUIL_BON    = 15.0   # std > 15 µV = bon contact électrode-peau
SEUIL_MOYEN  = 5.0    # std 5-15 µV = contact acceptable mais à améliorer
SEUIL_FAIBLE = 5.0    # std < 5 µV  = mauvais contact, repositionner

# Seuil de détection des artefacts
# Les segments dépassant ce seuil sont considérés comme des artefacts
# (mouvements brusques, décharges électroniques, etc.)
AMP_ARTEFACT = 150.0  # µV

# Canaux critiques pour l'imagerie motrice de la marche
# C3 = hémisphère gauche (contrôle membre droit)
# CZ = ligne médiane (contrôle membres inférieurs)
# C4 = hémisphère droit (contrôle membre gauche)
# Ces 3 canaux DOIVENT avoir un bon contact pour que l'acquisition soit valide
CANAUX_CRITIQUES = ['C3', 'CZ', 'C4']

# Bandes de fréquence EEG pertinentes pour l'imagerie motrice
# Mu (8-13 Hz) et Beta (13-30 Hz) sont les marqueurs principaux des ERD/ERS
BANDES = {
    'Delta (0-4Hz)'  : (0, 4,   'gray',   0.1),   # Sommeil profond
    'Theta (4-8Hz)'  : (4, 8,   'purple', 0.1),   # Somnolence
    'Mu (8-13Hz)'    : (8, 13,  'blue',   0.2),   # ⭐ Imagerie motrice
    'Beta (13-30Hz)' : (13, 30, 'green',  0.2),   # ⭐ Imagerie motrice
    'Gamma (30+Hz)'  : (30, 50, 'red',    0.1),   # Processus cognitifs
}


# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================

def load_csv(filepath):
    """
    Charge le fichier CSV exporté par le logiciel Kaptics.

    Le fichier CSV Kaptics contient :
    - Une colonne TimeStamp (en secondes)
    - Des colonnes Channel_X (Y) avec X = numéro, Y = nom de l'électrode
    - Des colonnes Accel_X, Accel_Y, Accel_Z pour l'accéléromètre
    - Une colonne Marker pour les événements Unity (0 = pas d'événement)

    Les valeurs EEG sont en nanovolts (nV) dans le fichier CSV.
    On les convertit en microvolts (µV) en divisant par 1000.
    Cette conversion est nécessaire car les seuils de qualité sont
    définis en µV (standard dans la littérature EEG).

    Args:
        filepath : chemin vers le fichier CSV Kaptics

    Returns:
        df_uv   : DataFrame avec les données en µV
        eeg_cols: liste des noms de colonnes EEG
        fs      : fréquence d'échantillonnage estimée (Hz)
    """
    print(f"\n{'='*60}")
    print(f"CHARGEMENT : {Path(filepath).name}")
    print(f"{'='*60}")

    # Lecture du CSV avec détection automatique du séparateur
    # (tabulation ou virgule selon la version du logiciel Kaptics)
    df = pd.read_csv(filepath, sep=None, engine='python')

    # Identification des colonnes EEG (contiennent "Channel" dans le nom)
    eeg_cols = [c for c in df.columns if 'Channel' in c]
    if not eeg_cols:
        print("❌ Aucun canal EEG trouvé !")
        print("   Vérifiez que le fichier CSV est bien exporté depuis Kaptics")
        print("   Les colonnes EEG doivent contenir 'Channel' dans leur nom")
        return None

    # Conversion nanovolts → microvolts (÷ 1000)
    # IMPORTANT : les seuils de qualité (SEUIL_BON, SEUIL_MOYEN) sont en µV
    df_uv = df.copy()
    df_uv[eeg_cols] = df[eeg_cols].apply(pd.to_numeric, errors='coerce') / 1000.0

    # Calcul de la fréquence d'échantillonnage depuis les timestamps
    # fs = nombre d'échantillons / durée totale
    if 'TimeStamp' in df.columns:
        df_uv['time'] = df['TimeStamp'] - df['TimeStamp'].iloc[0]
        fs = len(df) / df_uv['time'].max()
    else:
        # Valeur par défaut si pas de TimeStamp
        fs = 250.0
        df_uv['time'] = np.arange(len(df)) / fs

    print(f"  Durée         : {df_uv['time'].max():.1f} secondes")
    print(f"  Échantillons  : {len(df)}")
    print(f"  Fréquence     : {fs:.1f} Hz")
    print(f"  Canaux EEG    : {len(eeg_cols)}")

    return df_uv, eeg_cols, float(fs)


# ============================================================
# ANALYSE QUANTITATIVE
# ============================================================

def analyse_quantitative(df_uv, eeg_cols, fs):
    """
    Calcule les statistiques de qualité pour chaque canal EEG.

    Métriques calculées pour chaque canal :
    - Moyenne (µV) : doit être proche de 0 si le signal est centré
    - Écart-type std (µV) : indicateur principal de qualité du contact
      Un std élevé indique une bonne activité électrique capturée
    - Maximum absolu (µV) : détecte les artefacts de grande amplitude
    - Nombre d'artefacts : échantillons dépassant AMP_ARTEFACT µV
    - % trials valides : proportion d'échantillons sans artefact

    Verdict global :
    - BON : ≥ 8 canaux avec std ≥ SEUIL_BON ET canaux C3/CZ/C4 OK
    - ACCEPTABLE : ≥ 8 canaux avec std ≥ SEUIL_MOYEN ET C3/CZ/C4 OK
    - INSUFFISANT : sinon → repositionner le casque

    Args:
        df_uv   : DataFrame avec données en µV
        eeg_cols: liste des colonnes EEG
        fs      : fréquence d'échantillonnage

    Returns:
        results : liste de dicts avec les stats par canal
        verdict : tuple (texte, couleur, recommandation)
        n_bon, n_moyen, n_faible : compteurs par catégorie de qualité
    """
    print(f"\n{'='*60}")
    print("ANALYSE QUANTITATIVE")
    print(f"{'='*60}")

    results = []

    print(f"\n{'Canal':<20} {'Moy(µV)':>10} {'Std(µV)':>10} {'Max(µV)':>10} {'Artefacts':>10} {'%Valid':>8} {'Qualité'}")
    print("-"*90)

    for col in eeg_cols:
        vals = df_uv[col].values

        # Statistiques de base
        mean_v  = float(np.mean(vals))
        std_v   = float(np.std(vals))
        max_v   = float(np.abs(vals).max())

        # Détection des artefacts (amplitude > seuil)
        n_art   = int((np.abs(vals) > AMP_ARTEFACT).sum())
        pct_val = float(100 * (len(vals) - n_art) / len(vals))

        # Classification de la qualité selon le std
        # Le std est le meilleur indicateur car il mesure la variabilité
        # du signal — un signal actif a une grande variabilité
        if std_v >= SEUIL_BON:
            qualite = "✅ BON"
            score   = 3
        elif std_v >= SEUIL_MOYEN:
            qualite = "⚠️  MOYEN"
            score   = 2
        else:
            qualite = "❌ FAIBLE"
            score   = 1

        # Extraction du nom court de l'électrode depuis le nom de colonne
        # Format Kaptics : "Channel_1 (C3)" → on extrait "C3"
        nom = col.split('(')[1].replace(')', '') if '(' in col else col.replace('Channel','Ch')

        # Les canaux critiques (C3, CZ, C4) sont marqués d'une étoile
        is_critical = any(c in nom for c in CANAUX_CRITIQUES)
        marker = "⭐" if is_critical else "  "

        results.append({
            'col'        : col,
            'nom'        : nom,
            'mean'       : mean_v,
            'std'        : std_v,
            'max'        : max_v,
            'n_art'      : n_art,
            'pct_valid'  : pct_val,
            'qualite'    : qualite,
            'score'      : score,
            'is_critical': is_critical,
        })

        print(f"{marker}{col[:18]:<20} {mean_v:>10.2f} {std_v:>10.2f} {max_v:>10.2f} {n_art:>10} {pct_val:>7.1f}% {qualite}")

    # Comptage des canaux par catégorie de qualité
    n_bon    = sum(1 for r in results if r['score'] == 3)
    n_moyen  = sum(1 for r in results if r['score'] == 2)
    n_faible = sum(1 for r in results if r['score'] == 1)

    # Vérification spécifique des canaux critiques (C3, CZ, C4)
    # Ces 3 canaux DOIVENT être au moins MOYEN pour une acquisition valide
    crit_results = [r for r in results if r['is_critical']]
    crit_scores  = [r['score'] for r in crit_results]
    crit_ok      = all(s >= 2 for s in crit_scores)

    print(f"\n--- RÉSUMÉ ---")
    print(f"  Canaux BON    (std ≥ {SEUIL_BON}µV)  : {n_bon}/16")
    print(f"  Canaux MOYEN  (std ≥ {SEUIL_MOYEN}µV)   : {n_moyen}/16")
    print(f"  Canaux FAIBLE (std < {SEUIL_FAIBLE}µV)   : {n_faible}/16")
    print(f"  Canaux critiques OK (C3/CZ/C4) : {'✅ OUI' if crit_ok else '❌ NON'}")

    # Verdict global basé sur le nombre de canaux OK et les canaux critiques
    if n_bon >= 8 and crit_ok:
        verdict = ("✅ SIGNAL DE BONNE QUALITÉ", "green",
                   "Vous pouvez commencer l'acquisition.")
    elif n_bon + n_moyen >= 8 and crit_ok:
        verdict = ("⚠️  SIGNAL ACCEPTABLE", "orange",
                   "Contact à améliorer mais acquisition possible.")
    else:
        verdict = ("❌ SIGNAL INSUFFISANT", "red",
                   "Repositionner le casque avant de commencer !")

    print(f"\n{'='*60}")
    print(f"VERDICT : {verdict[0]}")
    print(f"         {verdict[2]}")
    print(f"{'='*60}")

    # Détection des marqueurs Unity (événements du protocole expérimental)
    marker_col = df_uv['Marker'] if 'Marker' in df_uv.columns else None
    n_markers  = int((marker_col != 0).sum()) if marker_col is not None else 0
    print(f"\n--- MARQUEURS ---")
    print(f"  Marqueurs détectés : {n_markers}")
    if n_markers == 0:
        print("  ℹ️  Normal si Unity n'est pas connecté pendant ce test")

    return results, verdict, n_bon, n_moyen, n_faible


# ============================================================
# ANALYSE GRAPHIQUE
# ============================================================

def analyse_graphique(df_uv, eeg_cols, results, verdict, fs, output_path):
    """
    Génère une figure d'analyse avec 7 panneaux.

    Panneaux générés :
    1. Signal brut tous canaux (décalés pour visibilité)
       → Vue d'ensemble de l'activité EEG sur l'ensemble du montage
    2. Canaux critiques C3, CZ, C4 zoomés
       → Qualité des canaux les plus importants pour l'imagerie motrice
    3. Barres d'écart-type par canal
       → Comparaison visuelle de la qualité de chaque électrode
    4. Camembert résumé qualité
       → Distribution des canaux BON/MOYEN/FAIBLE
    5. PSD (Densité Spectrale de Puissance) canaux C3/CZ/C4
       → Vérifie que les bandes mu (8-13Hz) et bêta (13-30Hz) sont présentes
    6. Accéléromètre
       → Détecte les mouvements de tête pendant l'enregistrement
    7. Verdict et recommandations
       → Résumé visuel avec les actions à entreprendre

    Args:
        df_uv       : DataFrame avec données en µV
        eeg_cols    : liste des colonnes EEG
        results     : liste de dicts avec stats par canal (de analyse_quantitative)
        verdict     : tuple (texte, couleur, recommandation)
        fs          : fréquence d'échantillonnage
        output_path : chemin de sauvegarde de la figure PNG
    """
    print(f"\n{'='*60}")
    print("GÉNÉRATION DES GRAPHIQUES")
    print(f"{'='*60}")

    t = df_uv['time'].values

    fig = plt.figure(figsize=(22, 28))
    fig.patch.set_facecolor('#F8F9FA')

    # Titre principal avec informations sur le fichier et le verdict
    verdict_color = verdict[1]
    fig.suptitle(
        f"Analyse Qualité Signal EEG — Casque Kaptics\n"
        f"Fichier : {Path(output_path).stem} | "
        f"Durée : {t.max():.1f}s | Fs : {fs:.0f} Hz | "
        f"Verdict : {verdict[0]}",
        fontsize=14, fontweight='bold', y=0.98,
        color='white',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#1F4E79', edgecolor='none')
    )

    gs = gridspec.GridSpec(5, 2, figure=fig,
                           hspace=0.45, wspace=0.3,
                           top=0.94, bottom=0.04)

    # Couleurs pour les canaux critiques (rouge=C3, bleu=CZ, vert=C4)
    crit_colors = {'C3': '#E74C3C', 'CZ': '#2980B9', 'C4': '#27AE60'}

    # ── PANNEAU 1 : Signal brut tous canaux ──────────────────────
    # Les signaux sont centrés (moyenne soustraite) et décalés
    # verticalement pour être visibles simultanément
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor('#FAFAFA')
    colors = plt.cm.tab20(np.linspace(0, 1, len(eeg_cols)))
    offset = 0
    for i, col in enumerate(eeg_cols):
        sig = df_uv[col].values - np.mean(df_uv[col].values)
        nom = col.split('(')[1].replace(')', '') if '(' in col else col
        ax1.plot(t, sig + offset, color=colors[i], linewidth=0.5,
                 label=nom, alpha=0.8)
        offset += 200  # Décalage de 200µV entre chaque canal
    ax1.set_title("Signal EEG brut — 16 canaux (centrés, décalés pour visibilité)",
                  fontsize=12, fontweight='bold', pad=8)
    ax1.set_xlabel("Temps (s)", fontsize=10)
    ax1.set_ylabel("Amplitude (µV)", fontsize=10)
    ax1.legend(loc='upper right', fontsize=6, ncol=4,
               framealpha=0.8, edgecolor='gray')
    ax1.grid(True, alpha=0.2, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ── PANNEAU 2 : Canaux critiques C3, CZ, C4 ─────────────────
    # Zoom sur les 3 canaux les plus importants pour l'imagerie motrice
    # avec indication de la qualité (std) dans la légende
    ax2 = fig.add_subplot(gs[1, :])
    ax2.set_facecolor('#FAFAFA')
    for col in eeg_cols:
        nom = col.split('(')[1].replace(')', '') if '(' in col else col
        if nom in CANAUX_CRITIQUES:
            sig = df_uv[col].values - np.mean(df_uv[col].values)
            color = crit_colors.get(nom, 'black')
            r = next(r for r in results if r['nom'] == nom)
            ax2.plot(t, sig, color=color, linewidth=1.2,
                     label=f"{nom} (std={r['std']:.2f}µV — {r['qualite']})",
                     alpha=0.9)
    ax2.set_title("⭐ Canaux critiques : C3, CZ, C4 — Cortex moteur primaire",
                  fontsize=12, fontweight='bold', pad=8)
    ax2.set_xlabel("Temps (s)", fontsize=10)
    ax2.set_ylabel("Amplitude (µV)", fontsize=10)
    ax2.legend(fontsize=10, framealpha=0.8)
    ax2.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    ax2.grid(True, alpha=0.2, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # ── PANNEAU 3 : Barres std par canal ────────────────────────
    # Visualisation de l'écart-type par canal avec code couleur
    # Les canaux critiques sont colorés différemment des autres
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.set_facecolor('#FAFAFA')
    noms   = [r['nom'] for r in results]
    stds   = [r['std'] for r in results]
    colors_bar = []
    for r in results:
        # Canaux critiques : tons plus saturés
        if r['is_critical']:
            colors_bar.append('#C0392B' if r['score'] == 1 else
                              '#E67E22' if r['score'] == 2 else '#27AE60')
        else:
            colors_bar.append('#E74C3C' if r['score'] == 1 else
                              '#F39C12' if r['score'] == 2 else '#2ECC71')

    bars = ax3.bar(noms, stds, color=colors_bar, edgecolor='white',
                   linewidth=0.8, zorder=3)
    # Lignes horizontales indiquant les seuils de qualité
    ax3.axhline(y=SEUIL_BON,   color='#27AE60', linestyle='--',
                linewidth=1.5, label=f'Seuil BON ({SEUIL_BON}µV)', zorder=4)
    ax3.axhline(y=SEUIL_MOYEN, color='#E67E22', linestyle='--',
                linewidth=1.5, label=f'Seuil MOYEN ({SEUIL_MOYEN}µV)', zorder=4)
    ax3.set_title("Écart-type par canal (µV)\nIndicateur de qualité du contact électrode-peau",
                  fontsize=11, fontweight='bold', pad=8)
    ax3.set_xlabel("Canal", fontsize=10)
    ax3.set_ylabel("STD (µV)", fontsize=10)
    ax3.tick_params(axis='x', rotation=45, labelsize=8)
    ax3.legend(fontsize=9, framealpha=0.8)
    ax3.grid(True, alpha=0.2, linestyle='--', axis='y', zorder=0)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # ── PANNEAU 4 : Camembert résumé qualité ────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.set_facecolor('#FAFAFA')
    n_bon    = sum(1 for r in results if r['score'] == 3)
    n_moyen  = sum(1 for r in results if r['score'] == 2)
    n_faible = sum(1 for r in results if r['score'] == 1)

    sizes  = [n_bon, n_moyen, n_faible]
    labels = [f"BON\n(≥{SEUIL_BON}µV)\n{n_bon} canaux",
              f"MOYEN\n({SEUIL_MOYEN}-{SEUIL_BON}µV)\n{n_moyen} canaux",
              f"FAIBLE\n(<{SEUIL_MOYEN}µV)\n{n_faible} canaux"]
    colors_pie = ['#2ECC71', '#F39C12', '#E74C3C']
    explode = [0.05, 0.05, 0.05]

    non_zero = [(s, l, c, e) for s, l, c, e in
                zip(sizes, labels, colors_pie, explode) if s > 0]
    if non_zero:
        s_, l_, c_, e_ = zip(*non_zero)
        wedges, texts, autotexts = ax4.pie(
            s_, labels=l_, colors=c_, explode=e_,
            autopct='%1.0f%%', startangle=90,
            textprops={'fontsize': 9},
            wedgeprops={'edgecolor': 'white', 'linewidth': 2}
        )
        for at in autotexts:
            at.set_fontsize(10)
            at.set_fontweight('bold')
    ax4.set_title(f"Résumé qualité — {len(results)} canaux",
                  fontsize=11, fontweight='bold', pad=8)

    # ── PANNEAU 5 : PSD canaux critiques ────────────────────────
    # La densité spectrale de puissance (PSD) montre la distribution
    # de l'énergie du signal par fréquence.
    # Pour un bon signal EEG d'imagerie motrice, on doit observer :
    # - Un pic dans la bande mu (8-13 Hz) au repos
    # - Une atténuation dans les bandes delta (<4 Hz) → pas de dérive lente
    ax5 = fig.add_subplot(gs[3, :])
    ax5.set_facecolor('#FAFAFA')

    for col in eeg_cols:
        nom = col.split('(')[1].replace(')', '') if '(' in col else col
        if nom in CANAUX_CRITIQUES:
            sig = df_uv[col].values - np.mean(df_uv[col].values)
            # Méthode de Welch : estimation robuste de la PSD
            # nperseg = taille de la fenêtre d'analyse (en échantillons)
            f, psd = welch(sig, fs=fs, nperseg=min(512, len(sig)//4))
            mask = f <= 50  # Limiter l'affichage à 0-50 Hz
            color = crit_colors.get(nom, 'black')
            ax5.semilogy(f[mask], psd[mask], color=color,
                        linewidth=2, label=nom, alpha=0.9)

    # Zones de couleur pour chaque bande fréquentielle
    band_info = [
        (0,  4,  '#BDC3C7', 0.3, 'Delta\n(0-4Hz)'),
        (4,  8,  '#D7BDE2', 0.3, 'Theta\n(4-8Hz)'),
        (8,  13, '#AED6F1', 0.4, 'Mu\n(8-13Hz)\n⭐MI'),
        (13, 30, '#A9DFBF', 0.4, 'Beta\n(13-30Hz)\n⭐MI'),
        (30, 50, '#F9E79F', 0.3, 'Gamma\n(30-50Hz)'),
    ]
    for (flo, fhi, col, alpha, label) in band_info:
        ax5.axvspan(flo, fhi, alpha=alpha, color=col, zorder=0)
        ax5.text((flo+fhi)/2, ax5.get_ylim()[0] if ax5.get_ylim()[0] > 0 else 1e-6,
                 label, ha='center', va='bottom', fontsize=7,
                 color='#2C3E50', fontweight='bold')

    ax5.set_title(
        "Densité Spectrale de Puissance (PSD) — Canaux C3, CZ, C4\n"
        "Les bandes Mu (8-13Hz) et Beta (13-30Hz) sont essentielles pour l'imagerie motrice",
        fontsize=11, fontweight='bold', pad=8)
    ax5.set_xlabel("Fréquence (Hz)", fontsize=10)
    ax5.set_ylabel("PSD (µV²/Hz)", fontsize=10)
    ax5.legend(fontsize=11, framealpha=0.8)
    ax5.grid(True, alpha=0.2, linestyle='--')
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)

    # ── PANNEAU 6 : Accéléromètre ───────────────────────────────
    # L'accéléromètre mesure les mouvements de tête du participant.
    # Un signal stable (peu de variation) indique que le participant
    # reste immobile, ce qui est essentiel pour éviter les artefacts
    # de mouvement dans le signal EEG.
    ax6 = fig.add_subplot(gs[4, 0])
    ax6.set_facecolor('#FAFAFA')
    accel_cols = [c for c in df_uv.columns if 'Accel' in c]
    accel_colors = ['#E74C3C', '#2ECC71', '#3498DB']
    for i, col in enumerate(accel_cols):
        ax6.plot(t, df_uv[col], color=accel_colors[i % 3],
                linewidth=0.8, label=col, alpha=0.8)
    ax6.set_title("Accéléromètre — Mouvements de tête\n(doit rester stable pendant l'acquisition)",
                  fontsize=11, fontweight='bold', pad=8)
    ax6.set_xlabel("Temps (s)", fontsize=10)
    ax6.set_ylabel("Accélération", fontsize=10)
    ax6.legend(fontsize=9, framealpha=0.8)
    ax6.grid(True, alpha=0.2, linestyle='--')
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)

    # ── PANNEAU 7 : Verdict et recommandations ──────────────────
    ax7 = fig.add_subplot(gs[4, 1])
    ax7.set_facecolor('#FAFAFA')
    ax7.axis('off')

    vc = {'green': '#27AE60', 'orange': '#E67E22', 'red': '#E74C3C'}[verdict[1]]

    # Box du verdict avec couleur selon la qualité
    ax7.add_patch(FancyBboxPatch((0.05, 0.70), 0.90, 0.25,
                                  boxstyle="round,pad=0.02",
                                  facecolor=vc, edgecolor='white',
                                  linewidth=2, transform=ax7.transAxes))
    ax7.text(0.5, 0.83, verdict[0], ha='center', va='center',
             transform=ax7.transAxes, fontsize=12, fontweight='bold',
             color='white')
    ax7.text(0.5, 0.74, verdict[2], ha='center', va='center',
             transform=ax7.transAxes, fontsize=9, color='white')

    # Recommandations pratiques pour améliorer le contact
    recs = [
        "• Humidifier les électrodes (eau saline)",
        "• Écarter les cheveux sous chaque électrode",
        "• Appuyer fermement sur le casque",
        "• C3, CZ, C4 doivent avoir std ≥ 10µV",
        "• Attendre 30s après pose pour stabilisation",
    ]
    ax7.text(0.5, 0.65, "Recommandations :", ha='center', va='center',
             transform=ax7.transAxes, fontsize=10, fontweight='bold',
             color='#2C3E50')
    for i, rec in enumerate(recs):
        ax7.text(0.05, 0.55 - i*0.10, rec, ha='left', va='center',
                transform=ax7.transAxes, fontsize=9, color='#2C3E50')

    # Affichage des std des canaux critiques en bas du panneau
    std_crit = [r['std'] for r in results if r['is_critical']]
    ax7.text(0.5, 0.05,
             f"C3={std_crit[0]:.1f}µV | CZ={std_crit[1]:.1f}µV | C4={std_crit[2]:.1f}µV"
             if len(std_crit) >= 3 else "Canaux critiques non trouvés",
             ha='center', va='center', transform=ax7.transAxes,
             fontsize=9, style='italic', color='#7F8C8D')

    # Sauvegarde de la figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='#F8F9FA')
    plt.close()
    print(f"  ✅ Graphique sauvegardé : {output_path}")


# ============================================================
# RAPPORT TEXTE
# ============================================================

def generer_rapport(results, verdict, fs, duree, n_samples, output_txt):
    """
    Génère un rapport texte complet avec toutes les statistiques.

    Ce rapport est sauvegardé à côté du fichier CSV analysé.
    Il peut être consulté rapidement pour connaître la qualité
    du signal sans avoir à rouvrir la figure PNG.

    Args:
        results    : liste de dicts avec stats par canal
        verdict    : tuple (texte, couleur, recommandation)
        fs         : fréquence d'échantillonnage
        duree      : durée de l'enregistrement en secondes
        n_samples  : nombre total d'échantillons
        output_txt : chemin de sauvegarde du rapport .txt
    """
    lines = []
    lines.append("="*70)
    lines.append("RAPPORT D'ANALYSE QUALITÉ — CASQUE EEG KAPTICS")
    lines.append("="*70)
    lines.append(f"\nDurée           : {duree:.1f} secondes")
    lines.append(f"Échantillons    : {n_samples}")
    lines.append(f"Fréquence       : {fs:.1f} Hz")
    lines.append(f"Canaux analysés : {len(results)}")

    lines.append(f"\n{'='*70}")
    lines.append("RÉSULTATS PAR CANAL")
    lines.append(f"{'='*70}")
    lines.append(f"{'Canal':<12} {'Std(µV)':>10} {'Max(µV)':>10} {'Artefacts':>10} {'Qualité'}")
    lines.append("-"*60)
    for r in results:
        marker = "⭐" if r['is_critical'] else "  "
        lines.append(f"{marker}{r['nom']:<12} {r['std']:>10.2f} {r['max']:>10.2f} {r['n_art']:>10} {r['qualite']}")

    lines.append(f"\n{'='*70}")
    lines.append("VERDICT FINAL")
    lines.append(f"{'='*70}")
    lines.append(f"{verdict[0]}")
    lines.append(f"{verdict[2]}")

    lines.append(f"\n{'='*70}")
    lines.append("RECOMMANDATIONS")
    lines.append(f"{'='*70}")
    lines.append("1. Humidifier légèrement les électrodes avec eau saline")
    lines.append("2. Écarter les cheveux sous chaque électrode (surtout C3, CZ, C4)")
    lines.append("3. Appuyer fermement le casque sur la tête")
    lines.append("4. Attendre 30 secondes après la pose pour stabilisation")
    lines.append("5. Ne commencer l'acquisition que si std(C3) > 10µV et std(C4) > 10µV")
    lines.append("6. Vérifier l'impédance dans le logiciel Kaptics si disponible")

    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  ✅ Rapport texte sauvegardé : {output_txt}")


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Point d'entrée principal du script.

    Workflow :
    1. Charger le fichier CSV Kaptics
    2. Calculer les statistiques par canal (analyse_quantitative)
    3. Générer la figure d'analyse (analyse_graphique)
    4. Sauvegarder le rapport texte (generer_rapport)
    """
    ap = argparse.ArgumentParser(
        description="Analyse qualité signal EEG Kaptics — À utiliser avant chaque acquisition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
    python analyse_kaptics.py --file session_01.csv
    python analyse_kaptics.py --file session_01.csv --out ./resultats/

Interprétation du verdict :
    ✅ BON      → Lancer l'acquisition
    ⚠️  MOYEN   → Acquisition possible mais améliorer le contact
    ❌ INSUFFISANT → Repositionner le casque et recommencer
        """
    )
    ap.add_argument(
        "--file",
        type=str,
        required=False,
        default="/Users/emma/Desktop/Projet fin de maitrise/Casque/test_casque/test_amaury_3.2/3.2.csv",
        help="Chemin vers le fichier CSV Kaptics (requis)"
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Dossier de sortie pour les fichiers générés (défaut : même dossier que le CSV)"
    )
    args = ap.parse_args()

    filepath = Path(args.file)
    if not filepath.exists():
        print(f"❌ Fichier non trouvé : {filepath}")
        print("   Vérifiez le chemin et relancez le script")
        return

    # Dossier de sortie (même dossier que le CSV par défaut)
    out_dir = Path(args.out) if args.out else filepath.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = filepath.stem

    # Étape 1 : Chargement des données
    result = load_csv(filepath)
    if result is None:
        return
    df_uv, eeg_cols, fs = result

    # Étape 2 : Analyse quantitative (stats par canal + verdict)
    results, verdict, n_bon, n_moyen, n_faible = analyse_quantitative(
        df_uv, eeg_cols, fs)

    # Étape 3 : Génération de la figure PNG
    png_path = out_dir / f"{stem}_analyse_qualite.png"
    analyse_graphique(df_uv, eeg_cols, results, verdict, fs, str(png_path))

    # Étape 4 : Sauvegarde du rapport texte
    txt_path = out_dir / f"{stem}_rapport.txt"
    generer_rapport(results, verdict, fs,
                    df_uv['time'].max(), len(df_uv), str(txt_path))

    # Résumé final dans le terminal
    print(f"\n{'='*60}")
    print("ANALYSE TERMINÉE !")
    print(f"{'='*60}")
    print(f"  Graphique : {png_path}")
    print(f"  Rapport   : {txt_path}")
    print(f"\n  VERDICT : {verdict[0]}")
    print(f"  {verdict[2]}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
