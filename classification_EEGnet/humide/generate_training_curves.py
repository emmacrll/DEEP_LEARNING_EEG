#!/usr/bin/env python3
"""
generate_training_curves.py — Courbes d'entraînement EEGNet et DeepConvNet
===========================================================================
Génère les courbes d'entraînement/validation et epochs d'early stopping
pour EEGNet et DeepConvNet sur 3 sujets représentatifs (Jour 1).

USAGE :
    python generate_training_curves.py

PRODUIT :
    ./courbes_entrainement/
    ├── courbes_EEGNet_SubjXX_Jour1.png       (gate + expert)
    ├── courbes_DeepConvNet_SubjXX_Jour1.png  (gate + expert)
    └── comparaison_early_stopping.png         (comparaison 3 sujets)
"""

import os, sys, glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

BASE_DIR   = "/Users/emma/Desktop/Code/code_pipeline"
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "courbes_entrainement")
NPZ_DIR1   = os.path.join(BASE_DIR, "npz_all_subjects_day1")
os.makedirs(OUTPUT_DIR, exist_ok=True)
sys.path.insert(0, MODELS_DIR)

from eegnet_4class import (
    set_seed, load_runs, zscore_runs_independent,
    build_gate_idle_vs_move, build_expert_rl,
    train_gate_with_curves, train_expert_with_curves,
    stratified_split_indices, fit_channel_zscore,
    apply_channel_zscore, EEGNet, SimpleDataset,
    WeightedDataset, predict_logits, balanced_accuracy,
    confusion_matrix as bcm, augment_expert_data,
)

SUBJECTS = ["Subj05", "Subj08", "Subj12"]
SEED     = 42
device   = torch.device("cpu")

# Hyperparamètres EEGNet retenus
GATE_PARAMS = dict(
    n_classes=2, val_split=0.2, seed_split=SEED,
    do_zscore=True, kernel_len=128, dropout=0.25,
    lr=5e-4, wd=1e-4, batch_size=64,
    epochs=200, patience=40, F1=8, D=2, F2=16,
)
EXPERT_PARAMS = dict(
    n_classes=3, val_split=0.2, seed_split=SEED,
    do_zscore=True, kernel_len=192, dropout=0.25,
    lr=5e-4, wd=1e-4, batch_size=64,
    epochs=200, patience=40, F1=8, D=2, F2=16,
    aug_factor=9,
)

# Hyperparamètres DeepConvNet retenus
DCN_DROPOUT  = 0.5
DCN_LR       = 1e-3
DCN_EPOCHS   = 200
DCN_PATIENCE = 30
DCN_BATCH    = 64


# ── Utilitaires ───────────────────────────────────────────────────────────────

def find_npz(subj, suffix):
    for f in glob.glob(os.path.join(NPZ_DIR1, f"*_{suffix}.npz")):
        if os.path.basename(f).lower().startswith(subj.lower()):
            return f
    return None


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    return [r.item() if hasattr(r, 'item') else r for r in d["runs"]]


def runs_to_array_dcn(runs_mu, runs_beta):
    X_list, y_list = [], []
    for r_mu, r_beta in zip(runs_mu, runs_beta):
        y = np.array(r_mu["y"]).astype(int)
        Xm = np.concatenate([
            np.concatenate([r_mu["X_move1"],   r_mu["X_move2"]],   axis=2),
            np.concatenate([r_beta["X_move1"], r_beta["X_move2"]], axis=2)], axis=1)
        Xi = np.concatenate([
            np.concatenate([r_mu["X_nomove1"],   r_mu["X_nomove2"]],   axis=2),
            np.concatenate([r_beta["X_nomove1"], r_beta["X_nomove2"]], axis=2)], axis=1)
        mask  = (y == 1) | (y == 2) | (y == 3)
        y_act = np.zeros_like(y[mask])
        y_act[y[mask] == 2] = 1
        y_act[y[mask] == 3] = 2
        X_list.append(Xm[mask]); y_list.append(y_act)
        X_list.append(Xi);       y_list.append(np.full(len(Xi), 3))
    if not X_list:
        return None, None
    return (np.concatenate(X_list).astype(np.float32),
            np.concatenate(y_list).astype(np.int64))


def augment_dcn(X, y, factor=3):
    Xa, ya = [X], [y]
    for _ in range(factor - 1):
        Xa += [X + np.random.normal(0, 0.01, X.shape).astype(np.float32),
               np.roll(X, np.random.randint(1, 10), axis=2).astype(np.float32)]
        ya += [y, y]
    return np.concatenate(Xa), np.concatenate(ya)


# ── DeepConvNet ───────────────────────────────────────────────────────────────

class DeepConvNet(nn.Module):
    def __init__(self, n_ch, n_cls, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 25, (1, 64), padding=(0, 32)),
            nn.Conv2d(25, 25, (n_ch, 1)),
            nn.ELU(), nn.MaxPool2d((1, 2)), nn.Dropout(dropout),
            nn.Conv2d(25, 50, (1, 64), padding=(0, 32)),
            nn.ELU(), nn.MaxPool2d((1, 2)), nn.Dropout(dropout),
            nn.Conv2d(50, 100, (1, 64), padding=(0, 32)),
            nn.ELU(), nn.MaxPool2d((1, 2)), nn.Dropout(dropout),
        )
        dummy  = torch.zeros(1, 1, n_ch, 500)
        n_flat = self.net(dummy).flatten(1).shape[1]
        self.fc = nn.Linear(n_flat, n_cls)

    def forward(self, x):
        return self.fc(self.net(x.unsqueeze(1)).flatten(1))


def train_dcn_curves(X, y, n_cls):
    """Entraîne DeepConvNet et retourne bal_acc train/val par epoch + early stop."""
    n_ch  = X.shape[1]
    model = DeepConvNet(n_ch, n_cls, DCN_DROPOUT).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=DCN_LR)
    fn    = nn.CrossEntropyLoss()

    idx   = np.random.permutation(len(X))
    n_val = max(1, int(0.2 * len(X)))
    X_tr  = torch.tensor(X[idx[n_val:]]).to(device)
    y_tr  = torch.tensor(y[idx[n_val:]]).to(device)
    X_val = torch.tensor(X[idx[:n_val]]).to(device)
    y_val = torch.tensor(y[idx[:n_val]]).to(device)

    dl = DataLoader(TensorDataset(X_tr, y_tr),
                    batch_size=DCN_BATCH, shuffle=True)

    tr_bals, val_bals = [], []
    best_bal, pat, best_state = -1.0, 0, None
    early_ep = DCN_EPOCHS

    for ep in range(DCN_EPOCHS):
        model.train()
        tr_pred, tr_true = [], []
        for xb, yb in dl:
            opt.zero_grad()
            out = model(xb)
            fn(out, yb).backward()
            opt.step()
            tr_pred.extend(out.argmax(1).cpu().numpy())
            tr_true.extend(yb.cpu().numpy())

        from sklearn.metrics import balanced_accuracy_score
        tr_bals.append(balanced_accuracy_score(tr_true, tr_pred))

        model.eval()
        with torch.no_grad():
            vp = model(X_val).argmax(1).cpu().numpy()
        val_bals.append(balanced_accuracy_score(y_val.cpu().numpy(), vp))

        if val_bals[-1] > best_bal + 1e-6:
            best_bal  = val_bals[-1]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= DCN_PATIENCE:
                early_ep = ep + 1
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, tr_bals, val_bals, early_ep


# ── Tracé des courbes ─────────────────────────────────────────────────────────

def plot_curves(tr_g, val_g, ep_g, tr_e, val_e, ep_e,
                subj, model_name, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        f"{model_name} — Courbes d'entraînement\n"
        f"Sujet : {subj} | Configuration : Jour 1",
        fontsize=13, fontweight='bold')

    for ax, tr, val, ep, title in [
        (axes[0], tr_g,  val_g,  ep_g,
         f"Gate (IDLE vs MOVE)\nEarly stop : epoch {ep_g}"),
        (axes[1], tr_e,  val_e,  ep_e,
         f"Expert (RIGHT / LEFT / WALK)\nEarly stop : epoch {ep_e}"),
    ]:
        eplist = list(range(1, len(tr) + 1))
        ax.plot(eplist, tr,  color='#2980B9', lw=1.5, label='Entraînement')
        ax.plot(eplist, val, color='#E74C3C', lw=1.5, label='Validation')
        ax.axvline(x=ep, color='#27AE60', ls='--',
                   lw=1.5, label=f'Early stop (ep {ep})')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel("Epoch"); ax.set_ylabel("Balanced Accuracy")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, ls='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ {out_path}")


# ── EEGNet ────────────────────────────────────────────────────────────────────

def run_eegnet_curves(subj):
    mu_p   = find_npz(subj, "Jour1_band0812_motor8_guessB")
    beta_p = find_npz(subj, "Jour1_band1330_motor8_guessB")
    if mu_p is None:
        print(f"  {subj}: fichiers introuvables"); return None

    runs_mu   = zscore_runs_independent(load_runs(mu_p))
    runs_beta = zscore_runs_independent(load_runs(beta_p))
    if len(runs_mu) < 2:
        return None

    # Split simple : tous sauf dernier = train
    Xg_l, yg_l, Xe_l, ye_l = [], [], [], []
    for rm, rb in zip(runs_mu[:-1], runs_beta[:-1]):
        Xg_i, yg_i = build_gate_idle_vs_move(
            rm, ["nomove1","nomove2"], ["move1","move4"], "time")
        Xe_i, ye_i = build_expert_rl(rm, rb, ["move1"], "time")
        Xg_l.append(Xg_i); yg_l.append(yg_i)
        Xe_l.append(Xe_i); ye_l.append(ye_i)

    Xg = np.concatenate(Xg_l); yg = np.concatenate(yg_l)
    Xe = np.concatenate(Xe_l); ye = np.concatenate(ye_l)

    print("    Gate...")
    _, _, _, tr_g, val_g, ep_g = train_gate_with_curves(
        Xg, yg, device, **GATE_PARAMS)

    print("    Expert...")
    _, _, _, tr_e, val_e, ep_e = train_expert_with_curves(
        Xe, ye, device, **EXPERT_PARAMS)

    out = os.path.join(OUTPUT_DIR, f"courbes_EEGNet_{subj}_Jour1.png")
    plot_curves(tr_g, val_g, ep_g, tr_e, val_e, ep_e, subj, "EEGNet", out)
    return ep_g, ep_e


# ── DeepConvNet ───────────────────────────────────────────────────────────────

def run_dcn_curves(subj):
    mu_p   = find_npz(subj, "Jour1_band0812_motor8_guessB")
    beta_p = find_npz(subj, "Jour1_band1330_motor8_guessB")
    if mu_p is None:
        print(f"  {subj}: fichiers introuvables"); return None

    runs_mu   = load_npz(mu_p)
    runs_beta = load_npz(beta_p)
    if len(runs_mu) < 2:
        return None

    X_tr, y_tr = runs_to_array_dcn(runs_mu[:-1], runs_beta[:-1])
    X_tr, y_tr = augment_dcn(X_tr, y_tr, 3)

    print("    Gate...")
    y_gate = (y_tr != 3).astype(np.int64)
    _, tr_g, val_g, ep_g = train_dcn_curves(X_tr, y_gate, 2)

    print("    Expert...")
    mask = y_tr != 3
    _, tr_e, val_e, ep_e = train_dcn_curves(X_tr[mask], y_tr[mask], 3)

    out = os.path.join(OUTPUT_DIR, f"courbes_DeepConvNet_{subj}_Jour1.png")
    plot_curves(tr_g, val_g, ep_g, tr_e, val_e, ep_e, subj, "DeepConvNet", out)
    return ep_g, ep_e


# ── Figure comparative ────────────────────────────────────────────────────────

def plot_comparison(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        "Epochs d'early stopping — EEGNet vs DeepConvNet\nJour 1 — Gate et Expert",
        fontsize=13, fontweight='bold')

    subjs = [r["subj"] for r in results]
    x = np.arange(len(subjs)); w = 0.35

    for ax, key, title in [
        (axes[0], "gate",   "Gate (IDLE vs MOVE)"),
        (axes[1], "expert", "Expert (RIGHT / LEFT / WALK)"),
    ]:
        eeg = [r[f"eeg_{key}"] for r in results]
        dcn = [r[f"dcn_{key}"] for r in results]
        ax.bar(x - w/2, eeg, w, label="EEGNet",      color='#2980B9', alpha=0.85)
        ax.bar(x + w/2, dcn, w, label="DeepConvNet", color='#E74C3C', alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(subjs, fontsize=10)
        ax.set_ylabel("Epoch d'early stopping"); ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, ls='--', axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "comparaison_early_stopping.png")
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ {out}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

#!/usr/bin/env python3
"""
run_dcn_subj12.py — Lance uniquement DeepConvNet pour Subj12, Jour 1
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_training_curves import (
    set_seed, SEED, run_dcn_curves
)
import numpy as np
import torch

def main():
    set_seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("--- Subj12 ---")
    print("  DeepConvNet...")
    try:
        result = run_dcn_curves("Subj12")
        if result is None:
            print("  ERREUR: fichiers introuvables ou données insuffisantes")
        else:
            ep_g, ep_e = result
            print(f"  DeepConvNet : gate ep{ep_g} / expert ep{ep_e}")
    except Exception as ex:
        print(f"  DeepConvNet ERREUR: {ex}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()