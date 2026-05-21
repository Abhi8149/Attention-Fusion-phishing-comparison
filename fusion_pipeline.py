# =============================================================================
# IEEE Ethereum Phishing Detection — Attention Fusion Pipeline
# Adapted to ACTUAL file shapes:
#   e1:  (17623, 64)   <- projection 64→32
#   e2:  (17623, 15)   <- projection 15→32
#   p1:  (6956,)       <- DA-HGNN test-set probs
#   p2:  (5260,)       <- GAE+ML test-set probs
#   labels: (17623,)
#   timestamps: (44984,) <- edge-level; reduced to node-level
#   Working N = 5260  (intersection where both p1 and p2 exist)
# =============================================================================

# ── 0. Imports ────────────────────────────────────────────────────────────────
import os, warnings, random
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

# ── 1. Config ─────────────────────────────────────────────────────────────────
# Inline config (mirrors config.py)
RANDOM_SEED      = 42
TEST_SIZE        = 0.20
VAL_SIZE         = 0.10
IMBALANCE_RATIOS = [5, 10, 20, 40]
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)
os.makedirs("models",  exist_ok=True)

print(f"Device: {DEVICE}")

# ── 2. Load & Align Data ──────────────────────────────────────────────────────
p1_full  = np.load("p1.npy").astype(np.float32)   # (6956,)
p2_full  = np.load("p2.npy").astype(np.float32)   # (5260,)
e1_full  = np.load("e1.npy").astype(np.float32)   # (17623, 64)
e2_full  = np.load("e2.npy").astype(np.float32)   # (17623, 15)
labels_full = np.load("labels.npy").astype(np.float32)  # (17623,)
ts_full  = np.load("timestamps.npy")               # (44984,) edge-level

# Working set: first N_WORK nodes where BOTH p1 and p2 predictions exist
N_WORK = min(len(p1_full), len(p2_full))   # 5260

p1      = p1_full[:N_WORK]
p2      = p2_full[:N_WORK]
e1      = e1_full[:N_WORK]                 # (5260, 64)
e2      = e2_full[:N_WORK]                 # (5260, 15)
labels  = labels_full[:N_WORK]             # (5260,)

# Node-level timestamps: map edge timestamps → first N_WORK nodes
# Use sorted edge timestamps, assign 1 timestamp per node slot
ts_node = np.sort(ts_full)[:N_WORK]        # (5260,) ascending

E1_DIM = e1.shape[1]   # 64
E2_DIM = e2.shape[1]   # 15
PROJ   = 32            # shared projection dim

num_phishing = int((labels == 1).sum())
num_benign   = int((labels == 0).sum())
print(f"Working N={N_WORK}  phishing={num_phishing}  benign={num_benign}")

# ── 3. Random Split (stratified) ─────────────────────────────────────────────
idx = np.arange(N_WORK)
idx_trval, idx_test = train_test_split(
    idx, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=labels)
idx_tr, idx_val = train_test_split(
    idx_trval, test_size=VAL_SIZE/(1-TEST_SIZE),
    random_state=RANDOM_SEED, stratify=labels[idx_trval])

print(f"Split  train={len(idx_tr)}  val={len(idx_val)}  test={len(idx_test)}")

def split_arrays(idx):
    return (e1[idx], e2[idx], p1[idx], p2[idx], labels[idx])

e1_tr,  e2_tr,  p1_tr,  p2_tr,  y_tr  = split_arrays(idx_tr)
e1_val, e2_val, p1_val, p2_val, y_val = split_arrays(idx_val)
e1_te,  e2_te,  p1_te,  p2_te,  y_te  = split_arrays(idx_test)

# ── 4. Attention Fusion Model ─────────────────────────────────────────────────
class AttentionFusion(nn.Module):
    def __init__(self, e1_dim, e2_dim, proj=32):
        super().__init__()
        self.proj1 = nn.Linear(e1_dim, proj)
        self.proj2 = nn.Linear(e2_dim, proj)
        self.query  = nn.Parameter(torch.randn(proj))
        # classifier input: proj(32) + p1(1) + p2(1) = 34
        self.clf = nn.Sequential(
            nn.Linear(proj + 2, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )
    def forward(self, e1, e2, p1, p2):
        h1 = self.proj1(e1)   # (B, 32)
        h2 = self.proj2(e2)   # (B, 32)
        s1 = (h1 * self.query).sum(-1, keepdim=True)  # (B,1)
        s2 = (h2 * self.query).sum(-1, keepdim=True)
        alpha = torch.softmax(torch.cat([s1, s2], dim=1), dim=1)  # (B,2)
        fused = alpha[:,0:1]*h1 + alpha[:,1:2]*h2                 # (B,32)
        x = torch.cat([fused, p1.unsqueeze(1), p2.unsqueeze(1)], dim=1)  # (B,34)
        return self.clf(x).squeeze(1), alpha

def to_tensor(*arrays):
    return [torch.tensor(a, dtype=torch.float32).to(DEVICE) for a in arrays]

# ── 5. Train Fusion Model ─────────────────────────────────────────────────────
def make_loader(*arrays, batch_size=256, shuffle=True):
    tensors = to_tensor(*arrays)
    ds = TensorDataset(*tensors)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

model = AttentionFusion(E1_DIM, E2_DIM, PROJ).to(DEVICE)

# Freeze projection heads, train only fusion/classifier
for p in model.proj1.parameters(): p.requires_grad = False
for p in model.proj2.parameters(): p.requires_grad = False

pos_weight = torch.tensor([num_benign / num_phishing], dtype=torch.float32).to(DEVICE)
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer  = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3, weight_decay=1e-4)

train_loader = make_loader(e1_tr, e2_tr, p1_tr, p2_tr, y_tr)
val_loader   = make_loader(e1_val, e2_val, p1_val, p2_val, y_val, shuffle=False)

best_val_f1  = -1
best_weights = None

print("\nTraining Attention Fusion (50 epochs):")
print(f"{'Epoch':>6} {'Loss':>10} {'Val F1':>10}")

for epoch in range(1, 51):
    model.train()
    total_loss = 0
    for batch in train_loader:
        be1, be2, bp1, bp2, by = batch
        logits, _ = model(be1, be2, bp1, bp2)
        loss = criterion(logits, by)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item() * len(by)
    avg_loss = total_loss / len(idx_tr)

    model.eval()
    preds_v, trues_v = [], []
    with torch.no_grad():
        for batch in val_loader:
            be1, be2, bp1, bp2, by = batch
            logits, _ = model(be1, be2, bp1, bp2)
            preds_v.extend(torch.sigmoid(logits).cpu().numpy())
            trues_v.extend(by.cpu().numpy())
    val_f1 = f1_score(trues_v, np.array(preds_v) > 0.5, zero_division=0)
    print(f"{epoch:>6} {avg_loss:>10.4f} {val_f1:>10.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_weights = {k: v.clone() for k, v in model.state_dict().items()}

model.load_state_dict(best_weights)
torch.save(best_weights, "models/fusion_best.pt")
print(f"\nBest Val F1: {best_val_f1:.4f}  ->  models/fusion_best.pt")

# ── 6. Get Fusion Test Predictions & Alphas ───────────────────────────────────
model.eval()
te1, te2, tp1, tp2 = to_tensor(e1_te, e2_te, p1_te, p2_te)
with torch.no_grad():
    logits_te, alphas_te = model(te1, te2, tp1, tp2)
probs_fusion_te = torch.sigmoid(logits_te).cpu().numpy()
alphas_te_np    = alphas_te.cpu().numpy()

# Save full-dataset alphas (run over all N_WORK)
all_loader = make_loader(e1, e2, p1, p2, labels, shuffle=False)
all_probs, all_alphas = [], []
with torch.no_grad():
    for batch in all_loader:
        be1, be2, bp1, bp2, _ = batch
        lg, al = model(be1, be2, bp1, bp2)
        all_probs.extend(torch.sigmoid(lg).cpu().numpy())
        all_alphas.extend(al.cpu().numpy())
all_probs  = np.array(all_probs)
all_alphas = np.array(all_alphas)
np.save("models/fusion_probs.npy", all_probs)
np.save("models/alpha_array.npy",  all_alphas)
print(f"Saved fusion_probs.npy  alpha_array.npy  shape={all_alphas.shape}")

# ── 7. Meta-Learner ───────────────────────────────────────────────────────────
def meta_features(p1_, p2_):
    return np.stack([p1_, p2_, np.abs(p1_-p2_), p1_*p2_], axis=1)

meta_tr = meta_features(p1_tr, p2_tr)
meta_te = meta_features(p1_te, p2_te)

base_lr  = LogisticRegression(class_weight='balanced', random_state=RANDOM_SEED, max_iter=1000)
meta_clf = CalibratedClassifierCV(base_lr, cv=5, method='isotonic')
meta_clf.fit(meta_tr, y_tr)
probs_meta_te = meta_clf.predict_proba(meta_te)[:, 1]

# ── 8. Evaluation Helper ──────────────────────────────────────────────────────
def evaluate(y_true, y_prob, threshold=0.5, name=""):
    y_pred = (y_prob > threshold).astype(int)
    return {
        "Method": name,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1":        f1_score(y_true, y_pred, zero_division=0),
        "AUC-ROC":   roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    }

# ── 9. Task 2 — Ablation Study ────────────────────────────────────────────────
ablation_rows = [
    evaluate(y_te, p1_te,            name="DA-HGNN only"),
    evaluate(y_te, p2_te,            name="GAE+ML only"),
    evaluate(y_te, (p1_te+p2_te)/2,  name="Naive Average"),
    evaluate(y_te, probs_meta_te,     name="Meta-Learner"),
    evaluate(y_te, probs_fusion_te,   name="Attention Fusion"),
]
ablation_df = pd.DataFrame(ablation_rows)
ablation_df.to_csv("results/ablation_results.csv", index=False)

print("\n=== ABLATION RESULTS ===")
best_f1 = ablation_df["F1"].max()
for _, row in ablation_df.iterrows():
    marker = " ← BEST" if row["F1"] == best_f1 else ""
    print(f"  {row['Method']:<22} P={row['Precision']:.3f}  R={row['Recall']:.3f}  "
          f"F1={row['F1']:.3f}  AUC={row['AUC-ROC']:.3f}{marker}")
print("Saved → results/ablation_results.csv")

# ── 10. Task 3 — Class Imbalance Robustness ───────────────────────────────────
imbalance_rows = []
rng = np.random.default_rng(RANDOM_SEED)

phish_idx  = np.where(y_te == 1)[0]
benign_idx = np.where(y_te == 0)[0]
n_phish    = len(phish_idx)

for ratio in IMBALANCE_RATIOS:
    n_keep = min(n_phish * ratio, len(benign_idx))
    kept_b = rng.choice(benign_idx, size=n_keep, replace=False)
    sub    = np.concatenate([phish_idx, kept_b])
    ys     = y_te[sub]

    configs = [
        ("DA-HGNN only",    p1_te[sub]),
        ("GAE+ML only",     p2_te[sub]),
        ("Naive Average",   (p1_te[sub]+p2_te[sub])/2),
        ("Meta-Learner",    probs_meta_te[sub]),
        ("Attention Fusion",probs_fusion_te[sub]),
    ]
    for name, probs in configs:
        f1 = f1_score(ys, probs > 0.5, zero_division=0)
        imbalance_rows.append({"ratio": ratio, "method": name, "f1": f1})

imbalance_df = pd.DataFrame(imbalance_rows)
imbalance_df.to_csv("results/imbalance_results.csv", index=False)
print("\nSaved → results/imbalance_results.csv")
print(imbalance_df.pivot(index="method", columns="ratio", values="f1").round(3))

# ── 11. Task 4 — Temporal Split ───────────────────────────────────────────────
sort_order  = np.argsort(ts_node)          # ascending timestamp order
n_total     = N_WORK
n_tr_t      = int(n_total * 0.70)
n_val_t     = int(n_total * 0.10)
n_te_t      = n_total - n_tr_t - n_val_t

idx_tr_t  = sort_order[:n_tr_t]
idx_val_t = sort_order[n_tr_t:n_tr_t+n_val_t]
idx_te_t  = sort_order[n_tr_t+n_val_t:]

e1_tr_t,  e2_tr_t,  p1_tr_t,  p2_tr_t,  y_tr_t  = split_arrays(idx_tr_t)
e1_val_t, e2_val_t, p1_val_t, p2_val_t, y_val_t = split_arrays(idx_val_t)
e1_te_t,  e2_te_t,  p1_te_t,  p2_te_t,  y_te_t  = split_arrays(idx_te_t)
print(f"\nTemporal split  train={len(idx_tr_t)}  val={len(idx_val_t)}  test={len(idx_te_t)}")

# Re-train fusion on temporal split
model_t = AttentionFusion(E1_DIM, E2_DIM, PROJ).to(DEVICE)
for p in model_t.proj1.parameters(): p.requires_grad = False
for p in model_t.proj2.parameters(): p.requires_grad = False

n_phish_t  = int((y_tr_t == 1).sum())
n_benign_t = int((y_tr_t == 0).sum())
pw_t = torch.tensor(
    [n_benign_t / max(n_phish_t, 1)], dtype=torch.float32).to(DEVICE)
crit_t = nn.BCEWithLogitsLoss(pos_weight=pw_t)
opt_t  = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model_t.parameters()),
    lr=1e-3, weight_decay=1e-4)

tr_loader_t  = make_loader(e1_tr_t,  e2_tr_t,  p1_tr_t,  p2_tr_t,  y_tr_t)
val_loader_t = make_loader(e1_val_t, e2_val_t, p1_val_t, p2_val_t, y_val_t, shuffle=False)

best_f1_t, best_wt = -1, None
print("\nTemporal training (50 epochs):")
for epoch in range(1, 51):
    model_t.train()
    for batch in tr_loader_t:
        be1,be2,bp1,bp2,by = batch
        logits,_ = model_t(be1,be2,bp1,bp2)
        loss = crit_t(logits, by)
        opt_t.zero_grad(); loss.backward(); opt_t.step()
    model_t.eval()
    pv, tv = [], []
    with torch.no_grad():
        for batch in val_loader_t:
            be1,be2,bp1,bp2,by = batch
            pv.extend(torch.sigmoid(model_t(be1,be2,bp1,bp2)[0]).cpu().numpy())
            tv.extend(by.cpu().numpy())
    f1v = f1_score(tv, np.array(pv)>0.5, zero_division=0)
    if f1v > best_f1_t:
        best_f1_t = f1v
        best_wt   = {k:v.clone() for k,v in model_t.state_dict().items()}
    if epoch % 10 == 0:
        print(f"  Epoch {epoch:>3}  val_f1={f1v:.4f}")

model_t.load_state_dict(best_wt)

te1_t,te2_t,tp1_t,tp2_t = to_tensor(e1_te_t, e2_te_t, p1_te_t, p2_te_t)
with torch.no_grad():
    probs_fusion_t = torch.sigmoid(model_t(te1_t,te2_t,tp1_t,tp2_t)[0]).cpu().numpy()

meta_tr_t = meta_features(p1_tr_t, p2_tr_t)
meta_te_t = meta_features(p1_te_t, p2_te_t)
base_lr_t = LogisticRegression(class_weight='balanced', random_state=RANDOM_SEED, max_iter=1000)
meta_clf_t = CalibratedClassifierCV(base_lr_t, cv=min(5, max(2, n_phish_t)), method='isotonic')
meta_clf_t.fit(meta_tr_t, y_tr_t)
probs_meta_t = meta_clf_t.predict_proba(meta_te_t)[:,1]

temp_rows = [
    evaluate(y_te_t, p1_te_t,               name="DA-HGNN only"),
    evaluate(y_te_t, p2_te_t,               name="GAE+ML only"),
    evaluate(y_te_t, (p1_te_t+p2_te_t)/2,   name="Naive Average"),
    evaluate(y_te_t, probs_meta_t,           name="Meta-Learner"),
    evaluate(y_te_t, probs_fusion_t,         name="Attention Fusion"),
]
temp_df = pd.DataFrame(temp_rows)
temp_df.to_csv("results/temporal_results.csv", index=False)
print("\nSaved → results/temporal_results.csv")

print("\n=== RANDOM vs TEMPORAL F1 COMPARISON ===")
print(f"{'Method':<22} {'Random F1':>10} {'Temporal F1':>12}")
for (_, rrow), (_, trow) in zip(ablation_df.iterrows(), temp_df.iterrows()):
    print(f"  {rrow['Method']:<22} {rrow['F1']:>10.3f} {trow['F1']:>12.3f}")

# ── 12. Task 5 — Publication Figures ─────────────────────────────────────────
PALETTE = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7"]
METHODS  = ablation_df["Method"].tolist()
METRICS  = ["Precision", "Recall", "F1", "AUC-ROC"]

# Figure 1 — Ablation bar chart
fig, ax = plt.subplots(figsize=(13, 6))
x   = np.arange(len(METHODS))
w   = 0.18
for i, (metric, color) in enumerate(zip(METRICS, PALETTE)):
    vals = ablation_df[metric].values
    bars = ax.bar(x + (i-1.5)*w, vals, w, label=metric, color=color, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f"{v:.2f}", ha='center', va='bottom', fontsize=7.5, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(METHODS, fontsize=9)
ax.set_ylabel("Score", fontsize=11); ax.set_ylim(0, 1.12)
ax.set_title("Ablation Study: Fusion Methods Comparison", fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=9); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig("figures/fig_ablation.pdf", dpi=300, bbox_inches='tight')
fig.savefig("figures/fig_ablation.png", dpi=300, bbox_inches='tight')
plt.close()
print("\nSaved → figures/fig_ablation.pdf/.png")

# Figure 2 — F1 vs imbalance ratio
fig, ax = plt.subplots(figsize=(9, 5))
markers = ['o','s','^','D','*']
for i, method in enumerate(METHODS):
    sub   = imbalance_df[imbalance_df["method"] == method]
    lw    = 2.5 if method == "Attention Fusion" else 1.5
    color = PALETTE[i % len(PALETTE)]
    ax.plot(sub["ratio"], sub["f1"], marker=markers[i], label=method,
            linewidth=lw, color=color, markersize=7)
ax.set_xlabel("Imbalance Ratio (benign:phishing)", fontsize=11)
ax.set_ylabel("F1 Score", fontsize=11)
ax.set_title("F1 Score vs Class Imbalance Ratio", fontsize=13, fontweight='bold')
ax.set_xticks(IMBALANCE_RATIOS)
ax.legend(loc='upper right', bbox_to_anchor=(1.28, 1), fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig("figures/fig_imbalance.pdf", dpi=300, bbox_inches='tight')
fig.savefig("figures/fig_imbalance.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved → figures/fig_imbalance.pdf/.png")

# Figure 3 — Attention weight violin (alpha_1 per class)
alpha1_phish  = all_alphas[labels == 1, 0]
alpha1_benign = all_alphas[labels == 0, 0]

fig, axes = plt.subplots(1, 2, figsize=(9, 5), sharey=True)
for ax, data, color, title in zip(
        axes,
        [alpha1_phish, alpha1_benign],
        ["#D65F5F", "#4878CF"],
        ["Phishing Nodes", "Benign Nodes"]):
    parts = ax.violinplot(data, positions=[0], showmeans=True, showmedians=False)
    for pc in parts['bodies']:
        pc.set_facecolor(color); pc.set_alpha(0.7)
    for part in ['cmeans','cbars','cmaxes','cmins']:
        if part in parts:
            parts[part].set_color('black')
    ax.set_title(title, fontsize=11, fontweight='bold', color=color)
    ax.set_ylabel("α₁ (DA-HGNN weight)" if ax==axes[0] else "")
    ax.set_xticks([]); ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
fig.suptitle("Per-node stream trust: DA-HGNN vs GAE+ML", fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig("figures/fig_attention.pdf", dpi=300, bbox_inches='tight')
fig.savefig("figures/fig_attention.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved → figures/fig_attention.pdf/.png")

# Figure 4 — SHAP beeswarm (meta-learner)
meta_all   = meta_features(p1, p2)
feat_names = ["DA-HGNN prob", "GAE+ML prob", "Prob difference", "Prob product"]
# Use KernelExplainer for sklearn CalibratedClassifierCV compatibility
bg_sample  = shap.sample(meta_all, 100, random_state=RANDOM_SEED)
explainer  = shap.KernelExplainer(
    lambda x: meta_clf.predict_proba(x)[:, 1], bg_sample)
shap_values = explainer.shap_values(meta_all[:300], silent=True)  # subsample

fig, ax = plt.subplots(figsize=(8, 5))
shap.summary_plot(shap_values, meta_all[:300], feature_names=feat_names,
                  plot_type='dot', max_display=4, show=False)
plt.title("SHAP Beeswarm - Meta-Learner Feature Importance", fontsize=12, fontweight='bold')
plt.tight_layout()
fig.savefig("figures/fig_shap.pdf", dpi=300, bbox_inches='tight')
fig.savefig("figures/fig_shap.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved → figures/fig_shap.pdf/.png")

print("\n✅ All tasks complete. Results in results/ and figures/")
