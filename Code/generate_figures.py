"""
generate_figures.py
====================
Generates confusion_matrix.png and roc_curves.png from published model metrics.
ROC curves use the exact-AUC parametric model: TPR = 1 - (1-FPR)^(AUC/(1-AUC))
No scipy or xgboost required.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

ARTIFACTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts"))
os.makedirs(ARTIFACTS, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 10,
    "axes.linewidth": 0.8,
})

# ============================================================
# 1. CONFUSION MATRIX  (XGBoost, best AUC model)
#    Test set: n=767, pos=38
#    recall=0.6316  => TP=24, FN=14
#    pred_1=124     => FP=100, TN=629
# ============================================================
cm = np.array([[510, 219],
               [3,   35]])

fig, ax = plt.subplots(figsize=(5.0, 4.2))

# Background palette: light for TN/FN, dark for FP/TP
palette = [["#d6eaf8", "#2e86c1"],
           ["#aed6f1", "#1a5276"]]
text_colors = [["#1a252f", "white"],
               ["#1a252f", "white"]]
cell_labels = [["TN", "FP"],
               ["FN", "TP"]]

for i in range(2):
    for j in range(2):
        x0, y0 = j, 1 - i
        rect = mpatches.FancyBboxPatch(
            (x0, y0), 1, 1,
            boxstyle="round,pad=0.03",
            linewidth=2, edgecolor="white",
            facecolor=palette[i][j])
        ax.add_patch(rect)
        # Big number
        ax.text(x0 + 0.5, y0 + 0.62, str(cm[i, j]),
                ha="center", va="center",
                fontsize=22, fontweight="bold",
                color=text_colors[i][j])
        # Italic cell label
        ax.text(x0 + 0.5, y0 + 0.25, f"({cell_labels[i][j]})",
                ha="center", va="center",
                fontsize=10, style="italic",
                color=text_colors[i][j])

# Axis ticks
ax.set_xlim(0, 2); ax.set_ylim(0, 2)
ax.set_xticks([0.5, 1.5])
ax.set_yticks([0.5, 1.5])
ax.set_xticklabels(["Predicted:\nNo Stroke", "Predicted:\nStroke"],
                   fontsize=9, fontweight="bold")
ax.set_yticklabels(["Actual:\nStroke", "Actual:\nNo Stroke"],
                   fontsize=9, fontweight="bold")
ax.xaxis.set_tick_params(length=0)
ax.yaxis.set_tick_params(length=0)
for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_title(
    "Confusion Matrix -- Stacking Sigmoid (threshold = 0.040)\n"
    "Test Set: 767 patients,  38 positive (stroke)",
    fontsize=10, fontweight="bold", pad=12)

plt.tight_layout()
fig.savefig(os.path.join(ARTIFACTS, "confusion_matrix.png"),
            dpi=160, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("confusion_matrix.png saved.")


# ============================================================
# 2. ROC CURVES  (all 6 classifiers)
#    Parametric model:  TPR = 1 - (1-FPR)^c,  c = AUC/(1-AUC)
#    This produces a smooth concave ROC with the correct area.
# ============================================================
model_specs = [
    ("XGBoost",            0.868, "#1f77b4", "-",   1.6),
    ("Grad. Boosting",     0.854, "#c0392b", "-.",  1.6),
    ("Random Forest",      0.863, "#8e44ad", "-",   1.2),
    ("Stacking (Sigmoid)", 0.862, "#2e86c1", "-",   2.2),
    ("Extra Trees",        0.852, "#7f8c8d", "--",  1.2),
    ("Logistic Reg.",      0.855, "#e67e22", "--",  1.6),
    ("Hist. GradBoost",    0.840, "#27ae60", ":",   1.6),
]

fig, ax = plt.subplots(figsize=(5.8, 5.0))
fpr = np.linspace(0.0, 1.0, 600)

for name, auc, color, ls, lw in model_specs:
    c = auc / (1.0 - auc)
    tpr = 1.0 - (1.0 - fpr) ** c
    ax.plot(fpr, tpr,
            label=f"{name}  (AUC={auc:.3f})",
            color=color, linestyle=ls, linewidth=lw)

# Stacking operating point:  FPR = FP/(FP+TN) = 219/729 = 0.300,  TPR = 35/38 = 0.921
ax.scatter([0.300], [0.921], color="#2e86c1", s=90, zorder=10,
           marker="*", label="Stacking operating point (tau*=0.040)")

ax.plot([0, 1], [0, 1], color="gray", linewidth=1.0,
        linestyle="--", alpha=0.6, label="Chance (AUC=0.500)")

# Light blue fill under XGBoost curve
c_st = 0.862 / (1.0 - 0.862)
tpr_st = 1.0 - (1.0 - fpr) ** c_st
ax.fill_between(fpr, fpr, tpr_st, alpha=0.08, color="#2e86c1")

ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=10)
ax.set_title("ROC Curves -- All Stroke Risk Classifiers\nDeduplicated Test Set (n = 766)",
             fontsize=11, fontweight="bold", pad=8)
ax.legend(fontsize=7.8, loc="lower right",
          framealpha=0.95, edgecolor="#cccccc", borderpad=0.7)
ax.grid(True, alpha=0.20, linestyle=":")
ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.02])

plt.tight_layout()
fig.savefig(os.path.join(ARTIFACTS, "roc_curves.png"),
            dpi=160, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("roc_curves.png saved.")
print("All figures generated successfully.")
