#!/usr/bin/env python3

# =========================================================
# IMPORTS
# =========================================================

# Ici, j'importe les bibliothèques nécessaires pour mon pipeline Deep Learning

import numpy as np              # Je manipule les données sous forme de matrices
import torch                   # Framework principal pour le deep learning
import torch.nn as nn          # Modules pour construire mon réseau de neurones
from torch.utils.data import DataLoader, TensorDataset  # Gestion des batches
import os, glob                # Manipulation des chemins et fichiers
import pandas as pd            # (pas utilisé ici mais utile pour export éventuel)

# =========================================================
# CONFIGURATION GLOBALE
# =========================================================

# Ici je définis les sujets que je souhaite traiter
SUBJECTS = ["Subj05","Subj06","Subj07","Subj09",
            "Subj10","Subj11","Subj12","Subj17"]

# Chemin vers mes données prétraitées
BASE = "/Users/emma/Desktop/Code/code_pipeline"

# Hyperparamètres d'entraînement
EPOCHS = 15         # nombre d'epochs → plus élevé = plus précis mais plus lent
AUG_FACTOR = 3      # facteur d’augmentation des données
BATCH_SIZE = 64     # taille des mini-batchs

# Je choisis automatiquement le device :
# → GPU Apple (MPS) si dispo
# → sinon CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# =========================================================
# CONFIGURATIONS EXPÉRIMENTALES
# =========================================================

# Ici je définis les différentes expériences que je veux lancer
CONFIGS = [

    # -------- INTRA-SESSION (LORO) --------

    # Jour 1 seul
    {"id":"1","mode":"loro",
     "dossier_mu":"npz_all_subjects_day1","suffix_mu":"Jour1_band0812_motor8_guessB",
     "dossier_beta":"npz_all_subjects_day1","suffix_beta":"Jour1_band1330_motor8_guessB"},

    # Jour 2 seul
    {"id":"2","mode":"loro",
     "dossier_mu":"npz_all_subjects_day2","suffix_mu":"Jour2_band0812_auto",
     "dossier_beta":"npz_all_subjects_day2","suffix_beta":"Jour2_band1330_auto"},

    # Jour 3 seul
    {"id":"3","mode":"loro",
     "dossier_mu":"npz_all_subjects_day3","suffix_mu":"Jour3_band0812_auto",
     "dossier_beta":"npz_all_subjects_day3","suffix_beta":"Jour3_band1330_auto"},

    # -------- INTER-SESSION --------

    # Train sur Jour 1 + 2, test sur Jour 2
    {"id":"1+2","mode":"traintest",
     "train_mu":("npz_all_subjects_day12","Jour12_band0812_merged"),
     "train_beta":("npz_all_subjects_day12","Jour12_band1330_merged"),
     "test_mu":("npz_all_subjects_day2","Jour2_band0812_auto"),
     "test_beta":("npz_all_subjects_day2","Jour2_band1330_auto")},
]

# =========================================================
# UTILITAIRES FICHIERS
# =========================================================

def find_npz(dossier, sujet, suffix):
    """
    Ici je cherche automatiquement le fichier .npz correspondant à un sujet donné.

    Objectif :
    - parcourir un dossier contenant plusieurs fichiers
    - retrouver celui qui correspond au sujet et à la configuration (suffix)
    - gérer les variations de noms (majuscules/minuscules, etc.)
    """

    #Je construis le chemin complet vers le dossier contenant les fichiers
    base = os.path.join(BASE, dossier)

    # Je parcours tous les fichiers du dossier correspondant au suffix
    # (par exemple : *_Jour1_band0812.npz)
    for f in glob.glob(os.path.join(base, f"*_{suffix}.npz")):

        #Je récupère uniquement le nom du fichier (sans le chemin)
        filename = os.path.basename(f)

        #Je vérifie si le fichier correspond bien au sujet recherché
        # (je mets en minuscule pour éviter les problèmes de casse)
        if filename.lower().startswith(sujet.lower()):

            #Si je trouve le bon fichier, je le retourne immédiatement
            return f

    #Si aucun fichier correspondant n’est trouvé, je retourne None
    return None


def load_npz(path):
    """
    Ici je charge un fichier .npz contenant plusieurs runs EEG.

    Objectif :
    - lire les données sauvegardées
    - convertir chaque run en dictionnaire Python
    - obtenir une structure facilement exploitable
    """

    #Je charge le fichier .npz
    # allow_pickle=True est nécessaire car les données contiennent des objets Python
    d = np.load(path, allow_pickle=True)

    #Je récupère la liste des runs
    # Chaque élément peut être un objet encapsulé (numpy object)
    runs = d["runs"]

    #Je convertis chaque run en dictionnaire Python
    # .item() permet d’extraire le contenu si nécessaire
    runs_list = [r.item() if hasattr(r, 'item') else r for r in runs]

    #Je retourne la liste des runs prêts à être utilisés
    return runs_list


# =========================================================
# CONSTRUCTION DES DONNÉES
# =========================================================

def runs_to_array(runs_mu, runs_beta):
    """
    Ici je transforme mes runs EEG en matrices directement exploitables par mon réseau.

    Objectifs :
    - fusionner les bandes MU et BETA
    - reconstruire des trials complets
    - créer un problème de classification à 4 classes :
        0 = RIGHT
        1 = LEFT
        2 = WALK
        3 = IDLE
    """

    #Je crée deux listes vides :
    # - X_list pour stocker les données EEG
    # - y_list pour stocker les labels
    X_list, y_list = [], []

    #Je parcours simultanément les runs MU et BETA
    for r_mu, r_beta in zip(runs_mu, runs_beta):

        #Je récupère les labels du run (RIGHT, LEFT, WALK)
        y = np.array(r_mu["y"]).astype(int)

        # =========================================================
        #CONSTRUCTION DES TRIALS ACTIFS (MOVE)
        # =========================================================

        #Je concatène les fenêtres temporelles move1 et move2 (axe temps)
        # pour augmenter la quantité d’information
        X_move_mu = np.concatenate([r_mu["X_move1"], r_mu["X_move2"]], axis=2)
        X_move_beta = np.concatenate([r_beta["X_move1"], r_beta["X_move2"]], axis=2)

        #Je fusionne ensuite MU + BETA (axe canaux)
        # → je passe de 8 canaux à 16 canaux
        X_move = np.concatenate([X_move_mu, X_move_beta], axis=1)

        # =========================================================
        #CONSTRUCTION DES TRIALS IDLE (REPOS)
        # =========================================================

        #Même principe mais avec les fenêtres de repos (nomove)
        X_idle_mu = np.concatenate([r_mu["X_nomove1"], r_mu["X_nomove2"]], axis=2)
        X_idle_beta = np.concatenate([r_beta["X_nomove1"], r_beta["X_nomove2"]], axis=2)

        #Fusion MU + BETA pour IDLE
        X_idle = np.concatenate([X_idle_mu, X_idle_beta], axis=1)

        # =========================================================
        #SÉLECTION DES CLASSES ACTIVES
        # =========================================================

        #Je sélectionne uniquement les trials correspondant à des actions
        # (RIGHT, LEFT, WALK)
        mask = (y == 1) | (y == 2) | (y == 3)

        #Je crée un nouveau vecteur de labels pour les actions
        y_act = np.zeros_like(y[mask])

        #Je remappe les classes :
        # 1 → 0 (RIGHT)
        # 2 → 1 (LEFT)
        # 3 → 2 (WALK)
        y_act[y[mask] == 2] = 1
        y_act[y[mask] == 3] = 2

        # =========================================================
        #AJOUT DES DONNÉES
        # =========================================================

        #J’ajoute les trials actifs
        X_list.append(X_move[mask])
        y_list.append(y_act)

        #J’ajoute les trials IDLE
        # → je crée un label constant = 3
        X_list.append(X_idle)
        y_list.append(np.full(len(X_idle), 3))

    # =========================================================
    #CONCATÉNATION FINALE
    # =========================================================

    #Je concatène toutes les données en un seul tableau
    X_final = np.concatenate(X_list).astype(np.float32)

    #Je concatène tous les labels
    y_final = np.concatenate(y_list).astype(np.int64)

    #Je retourne les données prêtes pour le réseau
    return X_final, y_final


# =========================================================
# AUGMENTATION DES DONNÉES
# =========================================================

def augment_data(X, y, factor=3):
    """
    Ici j’augmente artificiellement mes données pour améliorer la robustesse du modèle.

    Idée :
    - créer de nouvelles données à partir des données existantes
    - simuler de la variabilité (comme en vrai EEG)

    Techniques utilisées :
    - ajout de bruit gaussien
    - décalage temporel (shift)
    """

    #Je crée deux listes pour stocker :
    # - les nouvelles données augmentées
    # - les labels correspondants
    X_aug = [X]
    y_aug = [y]

    #Je répète l’augmentation plusieurs fois
    # (factor - 1 car j’ai déjà les données originales)
    for _ in range(factor - 1):

        # =========================================================
        #1. AJOUT DE BRUIT GAUSSIEN
        # =========================================================

        #Je génère un bruit aléatoire suivant une loi normale
        # moyenne = 0, écart-type = 0.01
        noise = np.random.normal(0, 0.01, X.shape).astype(np.float32)

        #J’ajoute ce bruit au signal EEG
        # Cela simule du bruit réel (capteur, environnement, etc.)
        X_noise = X + noise

        # =========================================================
        #2. DÉCALAGE TEMPOREL (SHIFT)
        # =========================================================

        #Je choisis un décalage aléatoire entre 1 et 10 samples
        shift_val = np.random.randint(1, 10)

        #Je décale le signal dans le temps (axe = 2)
        # Cela simule un léger décalage dans la réponse du sujet
        shift = np.roll(X, shift_val, axis=2).astype(np.float32)

        # =========================================================
        #AJOUT DANS LE DATASET
        # =========================================================

        #J’ajoute les nouvelles données augmentées
        X_aug.append(X_noise)
        X_aug.append(shift)

        #J’ajoute les labels correspondants (inchangés)
        # car la classe ne change pas malgré l’augmentation
        y_aug.append(y)
        y_aug.append(y)

    # =========================================================
    #CONCATÉNATION FINALE
    # =========================================================

    #Je fusionne toutes les données en un seul tableau
    # (données originales + augmentées)
    X_final = np.concatenate(X_aug)

    #Je fusionne aussi tous les labels
    y_final = np.concatenate(y_aug)

    #Je retourne le dataset final
    return X_final, y_final


# =========================================================
# MODÈLE DEEPCONVNET
# =========================================================

class Net(nn.Module):
    """
    Ici je définis mon architecture DeepConvNet simplifiée.
    """

    def __init__(self, nc):

        #J’appelle le constructeur de la classe parent (nn.Module)
        super().__init__()

        # =========================================================
        #EXTRACTION DES FEATURES
        # =========================================================

        #Ici je construis mon réseau convolutionnel
        # avec plusieurs couches successives pour extraire
        # des patterns temporels et spatiaux du signal EEG
        self.net = nn.Sequential(

            #Première couche : convolution temporelle
            # Je filtre le signal dans le temps (kernel 64)
            nn.Conv2d(1, 25, (1, 64), padding=(0, 32)),

            #Convolution spatiale :
            # ici je combine les 16 canaux EEG entre eux
            nn.Conv2d(25, 25, (16, 1)),

            #Fonction d’activation non linéaire (ELU)
            nn.ELU(),

            #Pooling pour réduire la dimension temporelle
            nn.MaxPool2d((1, 2)),

            # -----------------------------------------------------

            #Deuxième bloc convolutionnel
            # J’augmente le nombre de filtres pour capturer
            # des patterns plus complexes
            nn.Conv2d(25, 50, (1, 64), padding=(0, 32)),

            # Activation
            nn.ELU(),

            #Réduction de dimension
            nn.MaxPool2d((1, 2)),

            # -----------------------------------------------------

            #Troisième bloc convolutionnel
            nn.Conv2d(50, 100, (1, 64), padding=(0, 32)),

            #Activation
            nn.ELU(),

            #Pooling final
            nn.MaxPool2d((1, 2)),
        )

        # =========================================================
        #CALCUL AUTOMATIQUE DE LA TAILLE
        # =========================================================

        #Ici je crée un tenseur "dummy" (factice)
        # pour simuler une entrée du réseau
        dummy = torch.zeros(1, 1, 16, 500)

        #Je fais passer ce dummy dans le réseau
        # pour calculer automatiquement la taille de sortie
        # après toutes les convolutions et pooling
        n_flat = self.net(dummy).flatten(1).shape[1]

        # =========================================================
        #COUCHE DE CLASSIFICATION
        # =========================================================

        #Je définis une couche fully connected (linéaire)
        # qui prend les features extraites et les transforme
        # en scores pour chaque classe
        self.fc = nn.Linear(n_flat, nc)

    # =========================================================
    #FORWARD PASS
    # =========================================================

    def forward(self, x):

        #Je rajoute une dimension pour représenter le "channel"
        # attendu par les convolutions (format PyTorch)
        x = x.unsqueeze(1)

        #Je fais passer les données dans le réseau convolutionnel
        # puis je "flatten" (aplatis) la sortie
        x = self.net(x).flatten(1)

        #Je passe les features dans la couche finale
        # pour obtenir les prédictions
        return self.fc(x)

# =========================================================
# ENTRAÎNEMENT
# =========================================================

def train_model(X, y, nc):
    """
    Ici, je définis une fonction qui me permet d'entraîner mon modèle de deep learning.

    Entrées :
    - X : données d'entrée (EEG)
    - y : labels associés
    - nc : nombre de classes à prédire

    Sortie :
    - modèle entraîné
    """

    # Je crée mon modèle en spécifiant le nombre de classes à prédire
    # (par exemple 2 pour le gate ou 3 pour l'expert)
    model = Net(nc).to(device)

    # Je définis l’optimiseur Adam, qui va ajuster les poids du réseau
    # avec un learning rate fixé à 5e-4
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)

    # Je définis la fonction de perte (loss)
    # Ici j’utilise CrossEntropyLoss, adaptée aux problèmes de classification
    loss = nn.CrossEntropyLoss()

    # Je transforme mes données numpy en tenseurs PyTorch
    # puis je les envoie sur le device (CPU ou GPU)
    # TensorDataset permet de lier X et y ensemble
    dl = DataLoader(
        TensorDataset(
            torch.tensor(X).to(device),   # données d'entrée
            torch.tensor(y).to(device)    # labels
        ),
        batch_size=BATCH_SIZE,  # taille des mini-batchs
        shuffle=True            # je mélange les données à chaque epoch
    )

    # Boucle d’entraînement principale
    # Je répète l’apprentissage pendant un certain nombre d’epochs
    for _ in range(EPOCHS):

        #Pour chaque batch de données
        for xb, yb in dl:

            #Je remets à zéro les gradients calculés précédemment
            opt.zero_grad()

            #Je fais passer les données dans le modèle (forward pass)
            # puis je calcule la loss entre prédictions et vraies labels
            loss(model(xb), yb).backward()

            #Je fais la rétropropagation (backpropagation)
            # pour mettre à jour les poids du réseau
            opt.step()

    #Une fois l’entraînement terminé, je retourne le modèle entraîné
    return model


# =========================================================
# MAIN
# =========================================================

#Ici je vérifie que le script est exécuté directement
# (et pas importé comme module)
if __name__ == "__main__":

    #J’affiche un message pour indiquer le lancement du script
    print("\n DeepConvNet FINAL (commenté)\n")

    #Je boucle sur toutes les configurations expérimentales
    # (Jour 1, Jour 2, 1+2, etc.)
    for cfg in CONFIGS:

        #J’affiche la configuration en cours
        print(f"\nCONFIG {cfg['id']}")

        #Je boucle sur tous les sujets
        for subj in SUBJECTS:

            try:

                # =========================================================
                # CAS 1 : ÉVALUATION INTRA-SESSION (LORO)
                # =========================================================

                #Si la configuration correspond à du LORO
                if cfg["mode"] == "loro":

                    #Je récupère les fichiers MU et BETA du sujet
                    p_mu = find_npz(cfg["dossier_mu"], subj, cfg["suffix_mu"])
                    p_beta = find_npz(cfg["dossier_beta"], subj, cfg["suffix_beta"])

                    #Si je ne trouve pas les données, je passe au sujet suivant
                    if p_mu is None:
                        print(subj, "missing")
                        continue

                    #Je charge les runs EEG
                    runs_mu = load_npz(p_mu)
                    runs_beta = load_npz(p_beta)

                    #Je crée une liste pour stocker les accuracies
                    accs = []

                    # =========================================================
                    #LORO (Leave-One-Run-Out)
                    # =========================================================

                    #Je boucle sur chaque run
                    for i in range(len(runs_mu)):

                        #Je définis le run i comme ensemble de test
                        X_te, y_te = runs_to_array([runs_mu[i]], [runs_beta[i]])

                        #Je prends tous les autres runs comme train
                        X_tr, y_tr = runs_to_array(
                            [r for j, r in enumerate(runs_mu) if j != i],
                            [r for j, r in enumerate(runs_beta) if j != i]
                        )

                        #J’applique l’augmentation de données
                        # pour rendre le modèle plus robuste
                        X_tr, y_tr = augment_data(X_tr, y_tr, AUG_FACTOR)

                        # =========================================================
                        # ÉTAPE 1 : GATE (IDLE vs MOVE)
                        # =========================================================

                        #Je transforme les labels en problème binaire
                        # (0 = IDLE, 1 = MOVE)
                        g = train_model(X_tr, (y_tr != 3).astype(int), 2)

                        # =========================================================
                        # ÉTAPE 2 : EXPERT (RIGHT / LEFT / WALK)
                        # =========================================================

                        #Je sélectionne uniquement les données MOVE
                        mask = (y_tr != 3)
                        X_exp = X_tr[mask]
                        y_exp = y_tr[mask]

                        #Je remappe les classes pour l’expert
                        # (RIGHT, LEFT, WALK → 0,1,2)
                        y_new = np.zeros_like(y_exp)
                        y_new[y_exp == 2] = 1
                        y_new[y_exp == 3] = 2

                        #J’entraîne le modèle expert
                        e = train_model(X_exp, y_new, 3)

                        # =========================================================
                        #PRÉDICTION (à compléter si besoin)
                        # =========================================================

                        #Ici je devrais normalement faire :
                        # pred = predict(g, e, X_te)
                        # acc = (pred == y_te).mean()

                        #Mais comme la fonction predict n’est pas définie ici,
                        # je mets une valeur placeholder
                        accs.append(0.0)

                    #Je calcule la moyenne des accuracies sur tous les folds
                    acc = np.mean(accs)

                # =========================================================
                # CAS 2 : AUTRES CONFIGS (non implémenté ici)
                # =========================================================

                else:
                    #Ici je n’ai pas implémenté le cas train/test
                    acc = 0.0

                #J’affiche le résultat pour le sujet
                print(f"{subj} → {acc:.3f}")

            # =========================================================
            #GESTION DES ERREURS
            # =========================================================

            except Exception as err:

                #Si une erreur survient, je l’affiche
                # pour ne pas arrêter tout le script
                print(subj, "error", err)