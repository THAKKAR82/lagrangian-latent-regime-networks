"""Generate architecture diagram for LagrangianRegimeNet — horizontal layout."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent.parent / "reports" / "figures" / "architecture.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

C_INPUT  = "#E8F4FD"
C_ENC    = "#D4E6F1"
C_LATENT = "#D5F5E3"
C_INTEG  = "#FEF9E7"
C_HEAD   = "#F9EBEA"
C_OUT    = "#F4ECF7"
EDGE     = "#34495E"
ARROW    = "#2C3E50"

fig, ax = plt.subplots(figsize=(22, 9))
ax.set_xlim(0, 22)
ax.set_ylim(0, 9)
ax.axis("off")
fig.patch.set_facecolor("#FAFAFA")

CY = 4.5   # main-flow centre y


def fbox(ax, cx, cy, w, h, label, sublabel=None,
         fc="#FFFFFF", ec=EDGE, fontsize=9.5, subfontsize=7.8, radius=0.12):
    ax.add_patch(mpatches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=1.5, edgecolor=ec, facecolor=fc, zorder=2,
    ))
    if sublabel:
        ax.text(cx, cy + h * 0.16, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=EDGE, zorder=3)
        ax.text(cx, cy - h * 0.22, sublabel, ha="center", va="center",
                fontsize=subfontsize, color="#666666", style="italic", zorder=3)
    else:
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=EDGE, zorder=3)


def arr(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=1.5,
                                mutation_scale=13),
                zorder=3)


def seg(ax, x0, y0, x1, y1):
    ax.plot([x0, x1], [y0, y1], color=ARROW, lw=1.4, zorder=1)


# ── x / y layout ──────────────────────────────────────────────────────────────
X_IN   = 1.5
X_ENC  = 4.3
X_EMB  = 7.0
X_HEAD = 9.6    # z₀ / ż₀ heads share this x
X_MGR  = 10.65  # merge-vertical x (sits just left of integration)
X_INT  = 13.2   # integration centre  (left=10.95, right=15.45, width=4.5)
X_ZF   = 16.5   # z_final
X_CLS  = 18.8   # classifier
X_OUT  = 21.1   # output classes box

Y_Z0 = 6.5      # z₀ head y
Y_ZD = 2.5      # ż₀ head y

BW = 2.2    # standard box width
BH = 0.82   # standard box height
SW = 1.6    # small box width (embedding, z_final)
SH = 0.70
HW = 1.7    # head width
HH = 0.72   # head height
IW = 4.5    # integration width   (left=10.95)
IH = 4.2    # integration height  (top=6.6, bottom=2.4)

# 1 ── Input ───────────────────────────────────────────────────────────────────
fbox(ax, X_IN, CY, BW, BH, "Input window", r"(B, T=40, F)", fc=C_INPUT)
arr(ax, X_IN + BW / 2, CY, X_ENC - BW / 2, CY)

# 2 ── Conv1D Encoder ──────────────────────────────────────────────────────────
fbox(ax, X_ENC, CY, BW, BH, "Causal Conv1D Encoder",
     "conv → conv → last-step readout", fc=C_ENC)
arr(ax, X_ENC + BW / 2, CY, X_EMB - SW / 2, CY)

# 3 ── Shared embedding h ──────────────────────────────────────────────────────
fbox(ax, X_EMB, CY, SW, SH, "h  (shared embedding)",
     fc=C_ENC, fontsize=8.8, radius=0.10)

# fork right of embedding → up/down → heads
FX = X_EMB + SW / 2 + 0.25          # fork x
seg(ax, X_EMB + SW / 2, CY, FX, CY) # short stub
seg(ax, FX, CY, FX, Y_Z0)           # up
seg(ax, FX, CY, FX, Y_ZD)           # down
arr(ax, FX, Y_Z0, X_HEAD - HW / 2, Y_Z0)
arr(ax, FX, Y_ZD, X_HEAD - HW / 2, Y_ZD)

# 4 ── Latent heads ────────────────────────────────────────────────────────────
fbox(ax, X_HEAD, Y_Z0, HW, HH, "z₀ head", "initial position",
     fc=C_LATENT, fontsize=9, subfontsize=7.5)
fbox(ax, X_HEAD, Y_ZD, HW, HH, "ż₀ head", "initial velocity",
     fc=C_LATENT, fontsize=9, subfontsize=7.5)

# merge lines → integration
seg(ax, X_HEAD + HW / 2, Y_Z0, X_MGR, Y_Z0)   # top head → merge
seg(ax, X_HEAD + HW / 2, Y_ZD, X_MGR, Y_ZD)   # bot head → merge
seg(ax, X_MGR, Y_ZD, X_MGR, Y_Z0)             # vertical
arr(ax, X_MGR, CY, X_INT - IW / 2, CY)        # → integration

# 5 ── Integration block ───────────────────────────────────────────────────────
ax.add_patch(mpatches.FancyBboxPatch(
    (X_INT - IW / 2, CY - IH / 2), IW, IH,
    boxstyle="round,pad=0,rounding_size=0.15",
    linewidth=2.0, edgecolor="#C0922A", facecolor=C_INTEG, zorder=2,
))
ax.text(X_INT, CY + IH / 2 - 0.28, "Symplectic Euler Integration",
        ha="center", va="center", fontsize=9.5, fontweight="bold",
        color="#7D6608", zorder=3)
ax.text(X_INT, CY + IH / 2 - 0.60, "(n steps)",
        ha="center", va="center", fontsize=8.8, color="#7D6608", zorder=3)

# equations — start at CY+0.88, step 0.58, 4 lines → last at CY-0.86 (bottom at CY-2.1=2.4)
y_eq = CY + 0.88
for eq in [
    r"$q = \mathrm{coord\_net}(z)$",
    r"$V = \mathrm{potential\_net}(q)$",
    r"$\ddot{z} = -(\nabla_q V + \gamma\,\dot{z})\,/\,M_{\mathrm{diag}}$",
    r"$\dot{z} += \Delta t \cdot \ddot{z}, \quad z += \Delta t \cdot \dot{z}$",
]:
    ax.text(X_INT, y_eq, eq, ha="center", va="center",
            fontsize=8.5, color="#5D4037", zorder=3)
    y_eq -= 0.58

# arrow integration → z_final
arr(ax, X_INT + IW / 2, CY, X_ZF - SW / 2, CY)

# 6 ── z_final ─────────────────────────────────────────────────────────────────
fbox(ax, X_ZF, CY, SW, SH,
     r"$z_{\mathrm{final}}\;\in\;\mathbb{R}^{d}$",
     fc=C_LATENT, fontsize=9, radius=0.10)
arr(ax, X_ZF + SW / 2, CY, X_CLS - BW / 2, CY)

# 7 ── Classifier ──────────────────────────────────────────────────────────────
fbox(ax, X_CLS, CY, BW, BH, "Classifier head",
     "LayerNorm → Linear(·, 4)", fc=C_HEAD)

# 8 ── Output classes ──────────────────────────────────────────────────────────
OW, OH = 1.55, 3.5
arr(ax, X_CLS + BW / 2, CY, X_OUT - OW / 2, CY)
ax.add_patch(mpatches.FancyBboxPatch(
    (X_OUT - OW / 2, CY - OH / 2), OW, OH,
    boxstyle="round,pad=0,rounding_size=0.14",
    linewidth=1.8, edgecolor="#7D3C98", facecolor=C_OUT, zorder=2,
))
ax.text(X_OUT, CY + OH / 2 - 0.30, "Regime Logits",
        ha="center", va="center", fontsize=8.8, fontweight="bold",
        color="#4A235A", zorder=3)
ax.text(X_OUT, CY + OH / 2 - 0.58, "(softmax → 4)",
        ha="center", va="center", fontsize=7.8, color="#4A235A", zorder=3)

classes = ["Bull / Calm", "Bull / Stress", "Bear / Calm", "Bear / Stress"]
colors  = ["#1A8D4A",    "#B7950B",       "#922B21",     "#1A5276"]
for i, (cls, col) in enumerate(zip(classes, colors)):
    y = CY + OH / 2 - 0.95 - i * 0.60
    ax.text(X_OUT, y, cls, ha="center", va="center",
            fontsize=7.5, color=col, fontweight="bold", zorder=3)

# ── Legend ────────────────────────────────────────────────────────────────────
ax.text(0.22, 0.52,
        "γ: learned damping (scalar or vector)\n"
        "M: diagonal mass matrix (softplus)\n"
        "V: deep potential network",
        ha="left", va="center", fontsize=7.5, color="#555555",
        linespacing=1.7, zorder=4,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#CCCCCC", lw=1))

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(11.0, 8.55, "LagrangianRegimeNet — Architecture",
        ha="center", va="center", fontsize=14, fontweight="bold",
        color=EDGE, zorder=4)

plt.tight_layout(pad=0.3)
fig.savefig(OUT, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {OUT}")
