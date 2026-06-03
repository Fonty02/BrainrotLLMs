"""Publication figures for the brainrot activation-steering experiment.

Produces two groups of figures in plots/:
  RESULTS   (fig_results_*)  — the headline findings
  ABLATION  (fig_ablation_*) — supporting / sensitivity analyses
plus paper_tables.txt with the key numbers.

Metrics (all judged by a model judge):
  br  = brainrot                      (is_brainrot == 1)
  coh = coherent                      (coherence judge != INCOHERENT)
  cb  = COHERENT BRAINROT  = br & coh  ← headline success metric
  inc = incoherent (breakage)         = not coh

Coherence is recomputed from judge_coherence_raw because the stored is_coherent
column is corrupted on older runs (a parser bug scored "INCOHERENT" as coherent,
since the string contains the substring "COHERENT").
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = Path("steering_results")
OUT_DIR = Path("plots")
OUT_DIR.mkdir(exist_ok=True)
KEY = ["model_name", "technique", "layer_pct", "coefficient", "question_id"]

MODEL_SHORT = {
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "google/gemma-4-E2B-it": "Gemma-4-2B",
}
MODEL_ORDER = ["Qwen2.5-7B", "Llama-3.1-8B", "Gemma-4-2B"]
TECHS = ["dom", "pca", "aas"]
TECH_LABEL = {"dom": "DoM", "pca": "PCA", "aas": "AAS"}

# Colours
C_BR, C_CB, C_INC = "#d62728", "#2ca02c", "#8c8c8c"     # brainrot / coherent-brainrot / incoherent
TECH_C = {"dom": "#1f77b4", "pca": "#ff7f0e", "aas": "#2ca02c"}
MODEL_C = {"Qwen2.5-7B": "#4c72b0", "Llama-3.1-8B": "#dd8452", "Gemma-4-2B": "#55a868"}
LAYER_C = {25: "#e41a1c", 50: "#377eb8", 75: "#4daf4a"}

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 220, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 11, "legend.fontsize": 9, "axes.grid": True,
    "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False,
})


def _is_degenerate(text):
    t = str(text).strip()
    if len(t.split()) < 5:
        return True
    toks = t.split()
    if sum(toks[i] == toks[i - 1] for i in range(1, len(toks))) >= 5:
        return True
    a = re.findall(r"[a-zA-Z]", t)
    return bool(a) and len(set(c.lower() for c in a)) < 8


def load():
    dual = sorted(RESULTS_DIR.glob("*_dual_judged.csv"))
    files = dual if dual else [f for f in sorted(RESULTS_DIR.glob("*_judged.csv"))
                               if not f.name.endswith("_dual_judged.csv")]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(subset=KEY).reset_index(drop=True)
    df["model_short"] = df["model_name"].map(MODEL_SHORT)
    df["br"] = pd.to_numeric(df["is_brainrot"], errors="coerce").fillna(-1).astype(int).clip(lower=0)
    if "judge_coherence_raw" in df.columns:
        raw = df["judge_coherence_raw"].astype(str).str.upper()
        df["coh"] = np.where(raw.str.contains("INCOHERENT"), 0,
                             np.where(raw.str.contains("COHERENT"), 1, -1))
    elif "is_coherent" in df.columns:
        df["coh"] = pd.to_numeric(df["is_coherent"], errors="coerce").fillna(-1).astype(int)
    else:
        df["coh"] = (~df["response"].apply(_is_degenerate)).astype(int)
    df["cb"] = ((df["br"] == 1) & (df["coh"] == 1)).astype(int)
    df["inc"] = (df["coh"] == 0).astype(int)
    df["pos"] = df["coefficient"] > 0
    # ordinal "strength level" of the positive coefficients within each technique
    df["pos_level"] = np.nan
    for t in TECHS:
        pcs = sorted(df.loc[(df.technique == t) & df.pos, "coefficient"].unique())
        m = {c: i + 1 for i, c in enumerate(pcs)}
        idx = (df.technique == t) & df.pos
        df.loc[idx, "pos_level"] = df.loc[idx, "coefficient"].map(m)
    return df


def rate_ci(x):
    x = np.asarray(x, float)
    n = len(x)
    if n == 0:
        return 0.0, 0.0, 0.0, 0
    p = x.mean()
    se = np.sqrt(p * (1 - p) / n)
    return p, max(0, p - 1.96 * se), min(1, p + 1.96 * se), n


def agg(df, by, col):
    rows = []
    for kv, g in df.groupby(by):
        p, lo, hi, n = rate_ci(g[col])
        rows.append((*(kv if isinstance(kv, tuple) else (kv,)), p, lo, hi, n))
    cols = (by if isinstance(by, list) else [by]) + ["p", "lo", "hi", "n"]
    return pd.DataFrame(rows, columns=cols)


df = load()
POS = df[df.pos]
print(f"Loaded {len(df)} rows ({len(df)//600} configs). "
      f"brainrot={df.br.mean()*100:.1f}%  incoherent={df.inc.mean()*100:.1f}%  "
      f"coherent-brainrot={df.cb.mean()*100:.1f}%")


# ════════════════════════════════════════════════════════════════════
# RESULTS 1 — Brainrot / coherence trade-off (dose–response), per technique
# ════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
for ax, t in zip(axes, TECHS):
    sub = POS[POS.technique == t]
    for col, c, lab in [("br", C_BR, "Brainrot"),
                        ("cb", C_CB, "Coherent brainrot"),
                        ("inc", C_INC, "Incoherent (broken)")]:
        a = agg(sub, "coefficient", col).sort_values("coefficient")
        ax.plot(a.coefficient, a.p * 100, "-o", color=c, label=lab, lw=2, ms=5)
        ax.fill_between(a.coefficient, a.lo * 100, a.hi * 100, color=c, alpha=0.12)
    ax.set_title(f"{TECH_LABEL[t]}")
    ax.set_xlabel("Steering coefficient" + (" (degrees)" if t == "aas" else " (×natural diff)"))
axes[0].set_ylabel("Rate (%)")
axes[0].legend(loc="upper left", frameon=False)
fig.suptitle("Brainrot–coherence trade-off: a sweet spot at intermediate steering strength",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "fig_results_1_tradeoff.png")
plt.close(fig)
print("saved fig_results_1_tradeoff.png")


# ════════════════════════════════════════════════════════════════════
# RESULTS 2 — Coherent-brainrot rate by model × technique (positive coeffs)
# ════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8.5, 5))
x = np.arange(len(MODEL_ORDER))
w = 0.26
for i, t in enumerate(TECHS):
    ps, los, his = [], [], []
    for m in MODEL_ORDER:
        p, lo, hi, n = rate_ci(POS[(POS.model_short == m) & (POS.technique == t)]["cb"])
        ps.append(p * 100); los.append((p - lo) * 100); his.append((hi - p) * 100)
    ax.bar(x + (i - 1) * w, ps, w, color=TECH_C[t], label=TECH_LABEL[t],
           yerr=[los, his], capsize=3, error_kw=dict(lw=1))
ax.set_xticks(x); ax.set_xticklabels(MODEL_ORDER)
ax.set_ylabel("Coherent-brainrot rate (%)")
ax.set_title("Steering induces coherent brainrot — DoM and AAS work, PCA fails")
ax.legend(title="Technique", frameon=False)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig_results_2_model_technique.png")
plt.close(fig)
print("saved fig_results_2_model_technique.png")


# ════════════════════════════════════════════════════════════════════
# RESULTS 3 — Coherent-brainrot heatmap per model (technique×layer vs strength)
# ════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
levels = sorted(POS.pos_level.dropna().unique())
for ax, m in zip(axes, MODEL_ORDER):
    sub = POS[POS.model_short == m]
    rows = [f"{TECH_LABEL[t]} L{lp}" for t in TECHS for lp in (25, 50, 75)]
    mat = np.full((len(rows), len(levels)), np.nan)
    for ri, (t, lp) in enumerate([(t, lp) for t in TECHS for lp in (25, 50, 75)]):
        for ci, lv in enumerate(levels):
            g = sub[(sub.technique == t) & (sub.layer_pct == lp) & (sub.pos_level == lv)]
            if len(g):
                mat[ri, ci] = g["cb"].mean() * 100
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=60)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([f"L{int(l)}" for l in levels])
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=8)
    ax.set_xlabel("Steering strength (low→high)")
    ax.set_title(m)
    for ri in range(mat.shape[0]):
        for ci in range(mat.shape[1]):
            if not np.isnan(mat[ri, ci]):
                ax.text(ci, ri, f"{mat[ri, ci]:.0f}", ha="center", va="center",
                        color="white" if mat[ri, ci] < 38 else "black", fontsize=7)
    ax.grid(False)
fig.colorbar(im, ax=axes, label="Coherent-brainrot (%)", shrink=0.8, pad=0.01)
fig.suptitle("Where steering works: coherent-brainrot across configurations",
             fontsize=13, fontweight="bold")
fig.savefig(OUT_DIR / "fig_results_3_heatmap.png")
plt.close(fig)
print("saved fig_results_3_heatmap.png")


# ════════════════════════════════════════════════════════════════════
# ABLATION 1 — Layer depth
# ════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 5))
for t in TECHS:
    a = agg(POS[POS.technique == t], "layer_pct", "cb").sort_values("layer_pct")
    ax.errorbar(a.layer_pct, a.p * 100, yerr=[(a.p - a.lo) * 100, (a.hi - a.p) * 100],
                marker="o", color=TECH_C[t], label=TECH_LABEL[t], lw=2, capsize=3)
ax.set_xticks([25, 50, 75])
ax.set_xlabel("Steering layer (% of depth)")
ax.set_ylabel("Coherent-brainrot rate (%)")
ax.set_title("Ablation: deeper layers steer better")
ax.legend(title="Technique", frameon=False)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig_ablation_1_layer.png")
plt.close(fig)
print("saved fig_ablation_1_layer.png")


# ════════════════════════════════════════════════════════════════════
# ABLATION 2 — Direction & magnitude (full signed sweep): incoherence U-shape
# ════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
for ax, t in zip(axes, TECHS):
    sub = df[df.technique == t]
    for col, c, lab in [("br", C_BR, "Brainrot"), ("inc", C_INC, "Incoherent")]:
        a = agg(sub, "coefficient", col).sort_values("coefficient")
        ax.plot(a.coefficient, a.p * 100, "-o", color=c, label=lab, lw=2, ms=5)
    ax.axvline(0, color="k", lw=0.8, ls=":")
    ax.set_title(TECH_LABEL[t])
    ax.set_xlabel("Steering coefficient (signed)")
axes[0].set_ylabel("Rate (%)")
axes[0].legend(frameon=False, loc="upper center")
fig.suptitle("Ablation: brainrot appears only for positive steering; large magnitude (either sign) breaks coherence",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "fig_ablation_2_direction.png")
plt.close(fig)
print("saved fig_ablation_2_direction.png")


# ════════════════════════════════════════════════════════════════════
# ABLATION 3 — Question-category breakdown (coherent-brainrot, positive coeffs)
# ════════════════════════════════════════════════════════════════════
cats = sorted(POS["question_category"].dropna().unique())
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(cats))
w = 0.26
for i, m in enumerate(MODEL_ORDER):
    ps = [rate_ci(POS[(POS.model_short == m) & (POS.question_category == c)]["cb"])[0] * 100 for c in cats]
    ax.bar(x + (i - 1) * w, ps, w, color=MODEL_C[m], label=m)
ax.set_xticks(x)
ax.set_xticklabels([c.replace(" & ", "\n& ").replace(" / ", "\n/ ") for c in cats], fontsize=8)
ax.set_ylabel("Coherent-brainrot rate (%)")
ax.set_title("Ablation: brainrot induction by question category")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig_ablation_3_category.png")
plt.close(fig)
print("saved fig_ablation_3_category.png")


# ════════════════════════════════════════════════════════════════════
# ABLATION 4 — Natural style-shift norm by layer (method diagnostic)
# ════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 5))
norms = (df[df.technique.isin(["dom", "aas"])]
         .drop_duplicates(subset=["model_short", "layer_pct"])
         [["model_short", "layer_pct", "steering_vector_norm"]])
for m in MODEL_ORDER:
    d = norms[norms.model_short == m].sort_values("layer_pct")
    ax.plot(d.layer_pct, d.steering_vector_norm, "-o", color=MODEL_C[m], label=m, lw=2)
ax.set_xticks([25, 50, 75])
ax.set_xlabel("Steering layer (% of depth)")
ax.set_ylabel("Natural style-shift norm ‖μ⁺−μ⁻‖")
ax.set_title("Ablation: per-layer style-shift magnitude (motivates norm-relative coeffs)")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig_ablation_4_norms.png")
plt.close(fig)
print("saved fig_ablation_4_norms.png")


# ════════════════════════════════════════════════════════════════════
# Paper tables (text)
# ════════════════════════════════════════════════════════════════════
lines = []
def w(s=""): lines.append(s)

w("KEY NUMBERS FOR PAPER")
w("=" * 60)
w(f"Total generations (deduped): {len(df)}  | configs: {len(df)//600}")
w(f"Overall: brainrot={df.br.mean()*100:.1f}%  incoherent={df.inc.mean()*100:.1f}%  "
  f"coherent-brainrot={df.cb.mean()*100:.1f}%")
w("")
w("Coherent-brainrot by model (positive coeffs):")
for m in MODEL_ORDER:
    p, lo, hi, n = rate_ci(POS[POS.model_short == m]["cb"])
    w(f"  {m:14s} {p*100:5.1f}%  [{lo*100:.1f}-{hi*100:.1f}]  (n={n})")
w("Coherent-brainrot by technique (positive coeffs):")
for t in TECHS:
    p, lo, hi, n = rate_ci(POS[POS.technique == t]["cb"])
    w(f"  {TECH_LABEL[t]:5s} {p*100:5.1f}%  [{lo*100:.1f}-{hi*100:.1f}]  (n={n})")
w("Coherent-brainrot by layer (positive coeffs):")
for lp in (25, 50, 75):
    p, lo, hi, n = rate_ci(POS[POS.layer_pct == lp]["cb"])
    w(f"  L{lp}%  {p*100:5.1f}%  [{lo*100:.1f}-{hi*100:.1f}]  (n={n})")
w("")
w("TOP 15 configurations by coherent-brainrot rate:")
cfg = (POS.groupby(["model_short", "technique", "layer_pct", "coefficient"])
       .agg(cb=("cb", "mean"), br=("br", "mean"), inc=("inc", "mean"), n=("cb", "size"))
       .reset_index().sort_values("cb", ascending=False))
w(f"  {'model':14s} {'tech':4s} {'layer':5s} {'coeff':>6s} {'cb%':>6s} {'br%':>6s} {'inc%':>6s}")
for _, r in cfg.head(15).iterrows():
    w(f"  {r.model_short:14s} {TECH_LABEL[r.technique]:4s} L{int(r.layer_pct):<4d} "
      f"{r.coefficient:>6g} {r.cb*100:6.1f} {r.br*100:6.1f} {r.inc*100:6.1f}")
(Path("paper_tables.txt")).write_text("\n".join(lines), encoding="utf-8")
print("saved paper_tables.txt")
print("\nAll figures in", OUT_DIR.resolve())
