"""Plot steering experiment results from judged CSVs.

Degenerate (garbled/repetitive) responses are filtered out before analysis.
"""

from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

RESULTS_DIR = Path("steering_results")
OUT_DIR = Path("plots")
OUT_DIR.mkdir(exist_ok=True)

MODEL_SHORT = {
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "google/gemma-4-E2B-it": "Gemma-4-2B",
}
TECHNIQUE_LABEL = {"dom": "DoM", "pca": "PCA", "aas": "AAS"}


def is_degenerate(text: str) -> bool:
    t = str(text).strip()
    if len(t) < 20:
        return True
    tokens = t.split()
    if len(tokens) < 5:
        return True
    runs = 0
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i - 1]:
            runs += 1
    if runs >= 5:
        return True
    alpha_chars = re.findall(r"[a-zA-Z]", t)
    if len(alpha_chars) > 0:
        unique_count = len(set(c.lower() for c in alpha_chars))
        if unique_count < 8:
            return True
    return False


# ── Load all judged CSVs (prefer dual-judged: brainrot + coherence) ──
dual_files = sorted(RESULTS_DIR.glob("*_dual_judged.csv"))
if dual_files:
    files = dual_files
else:
    files = [f for f in sorted(RESULTS_DIR.glob("*_judged.csv"))
             if not f.name.endswith("_dual_judged.csv")]
dfs = [pd.read_csv(f) for f in files]

df_all = pd.concat(dfs, ignore_index=True)
df_all["model_short"] = df_all["model_name"].map(MODEL_SHORT)
df_all["coeff_sign"] = np.where(df_all["coefficient"] > 0, "positive", "negative")
df_all["is_brainrot"] = pd.to_numeric(df_all["is_brainrot"], errors="coerce").fillna(-1).astype(int)

# Breakage gate: judge coherence if available, else text heuristic.
if "is_coherent" in df_all.columns:
    df_all["is_coherent"] = pd.to_numeric(df_all["is_coherent"], errors="coerce").fillna(-1).astype(int)
    df_all["degenerate"] = df_all["is_coherent"] != 1
else:
    df_all["degenerate"] = df_all["response"].apply(is_degenerate)
df_clean = df_all[~df_all["degenerate"]]

print(f"Loaded {len(df_all)} rows from {len(dfs)} files")
print(f"Degenerate filtered: {df_all['degenerate'].sum()} rows removed")
print(f"Clean rows: {len(df_clean)}")
print(f"Models: {df_clean['model_short'].unique().tolist()}")
print(f"Techniques: {df_clean['technique'].unique().tolist()}")
print(f"Coefficients: {sorted(df_clean['coefficient'].unique())}")


# ── Helper: compute brainrot rate with binomial CI ─────────────────
def brainrot_rate(group):
    n = len(group)
    k = (group["is_brainrot"] == 1).sum()
    rate = k / n if n > 0 else 0
    se = np.sqrt(rate * (1 - rate) / n) if n > 0 else 0
    ci_low = max(0, rate - 1.96 * se)
    ci_high = min(1, rate + 1.96 * se)
    return pd.Series({"brainrot_rate": rate, "ci_low": ci_low, "ci_high": ci_high, "n": n})


# ── Plot 1: Dose-response curves (per model, per technique) ───────
fig, axes = plt.subplots(3, 3, figsize=(18, 14), sharey=True)

for col, technique in enumerate(["dom", "pca", "aas"]):
    for row, (model_full, model_short) in enumerate(MODEL_SHORT.items()):
        ax = axes[row][col]
        subset = df_clean[(df_clean["model_name"] == model_full) & (df_clean["technique"] == technique)]
        if subset.empty:
            ax.set_title(f"{model_short} × {TECHNIQUE_LABEL[technique]}\n(no clean data)")
            continue

        agg = subset.groupby(["layer_pct", "coefficient"]).apply(brainrot_rate).reset_index()

        for lpct, color in [(25, "#e41a1c"), (50, "#377eb8"), (75, "#4daf4a")]:
            d = agg[agg["layer_pct"] == lpct].sort_values("coefficient")
            if d.empty:
                continue
            ax.errorbar(d["coefficient"], d["brainrot_rate"],
                        yerr=[d["brainrot_rate"] - d["ci_low"], d["ci_high"] - d["brainrot_rate"]],
                        marker="o", color=color, label=f"L{lpct}%", capsize=3, linewidth=1.5)

        ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
        ax.set_title(f"{model_short} × {TECHNIQUE_LABEL[technique]}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Coefficient")
        if col == 0:
            ax.set_ylabel("Brainrot Rate")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

fig.suptitle("Dose-Response: Brainrot Rate vs Steering Coefficient (clean)", fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(OUT_DIR / "dose_response.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved dose_response.png")

# ── Plot 2: Heatmap — brainrot rate by technique × layer × coeff ──
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for idx, model_full in enumerate(MODEL_SHORT):
    model_short = MODEL_SHORT[model_full]
    ax = axes[idx]
    subset = df_clean[df_clean["model_name"] == model_full]
    agg = subset.groupby(["technique", "layer_pct", "coefficient"]).apply(brainrot_rate).reset_index()

    agg["config"] = agg["technique"].map(TECHNIQUE_LABEL) + " L" + agg["layer_pct"].astype(str) + "%"
    pivot = agg.pivot_table(index="config", columns="coefficient", values="brainrot_rate")
    coeffs = sorted(agg["coefficient"].unique())
    pivot = pivot.reindex(columns=coeffs)

    if pivot.empty:
        ax.set_title(f"{model_short}\n(no clean data)")
        continue

    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(coeffs)))
    ax.set_xticklabels([f"{c:+.0f}" for c in coeffs])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title(model_short, fontsize=12, fontweight="bold")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iloc[i, j]
            if pd.notna(v):
                color = "white" if v < 0.4 or v > 0.6 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color=color)

    plt.colorbar(im, ax=ax, label="Brainrot Rate", shrink=0.85)

fig.suptitle("Brainrot Rate Heatmap by Configuration (clean only)", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved heatmap.png")

# ── Plot 3: Positive vs negative coefficients summary ─────────────
fig, ax = plt.subplots(figsize=(12, 6))

summary = df_clean.groupby(["model_short", "technique", "layer_pct", "coeff_sign"]).apply(brainrot_rate).reset_index()

configs = sorted(summary[["model_short", "technique", "layer_pct"]].drop_duplicates().itertuples(index=False),
                 key=lambda x: (x[0], x[1], x[2]))

x_labels = [f"{m}\n{TECHNIQUE_LABEL[t]} L{l}%" for m, t, l in configs]
x = np.arange(len(configs))
width = 0.35

pos_rates, neg_rates = [], []

for m, t, l in configs:
    r_pos = summary[(summary["model_short"] == m) & (summary["technique"] == t) &
                     (summary["layer_pct"] == l) & (summary["coeff_sign"] == "positive")]
    r_neg = summary[(summary["model_short"] == m) & (summary["technique"] == t) &
                     (summary["layer_pct"] == l) & (summary["coeff_sign"] == "negative")]
    pos_rates.append(r_pos["brainrot_rate"].values[0] if not r_pos.empty else 0)
    neg_rates.append(r_neg["brainrot_rate"].values[0] if not r_neg.empty else 0)

ax.bar(x - width/2, pos_rates, width, label="Positive coeff (toward brainrot)", color="#d7191c")
ax.bar(x + width/2, neg_rates, width, label="Negative coeff (away from brainrot)", color="#2b83ba")

ax.axhline(0, color="gray", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=7, rotation=45, ha="right")
ax.set_ylabel("Brainrot Rate")
ax.set_title("Brainrot Rate: Positive vs Negative Coefficients (clean)", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis="y")

fig.tight_layout()
fig.savefig(OUT_DIR / "positive_vs_negative.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved positive_vs_negative.png")

# ── Plot 4: Category breakdown ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

categories = sorted(df_clean["question_category"].dropna().unique())

for idx, model_full in enumerate(MODEL_SHORT):
    model_short = MODEL_SHORT[model_full]
    ax = axes[idx]
    subset = df_clean[df_clean["model_name"] == model_full]

    cat_data = []
    for cat in categories:
        for coeff in sorted(subset["coefficient"].unique()):
            d = subset[(subset["question_category"] == cat) & (subset["coefficient"] == coeff)]
            r = brainrot_rate(d)
            if r["n"] > 0:
                cat_data.append({"category": cat, "coefficient": coeff, **r.to_dict()})

    cat_df = pd.DataFrame(cat_data)
    if cat_df.empty:
        ax.set_title(f"{model_short}\n(no clean data)")
        continue

    for cat, marker in zip(categories, ["o", "s", "D", "^", "v"]):
        d = cat_df[cat_df["category"] == cat].sort_values("coefficient")
        if d.empty:
            continue
        ax.plot(d["coefficient"], d["brainrot_rate"], marker=marker, label=cat, linewidth=1.5)

    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    ax.set_title(model_short, fontsize=12, fontweight="bold")
    ax.set_xlabel("Coefficient")
    if idx == 0:
        ax.set_ylabel("Brainrot Rate")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle("Brainrot Rate by Question Category (clean)", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "by_category.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved by_category.png")

# ── Plot 5: Overall model × technique effectiveness summary ───────
fig, ax = plt.subplots(figsize=(10, 6))

summary2 = df_clean.groupby(["model_short", "technique", "coefficient"]).apply(brainrot_rate).reset_index()

models = sorted(summary2["model_short"].unique())
techniques = ["dom", "pca", "aas"]
colors = {"dom": "#e41a1c", "pca": "#377eb8", "aas": "#4daf4a"}

x = np.arange(len(models))
width = 0.25

for i, tech in enumerate(techniques):
    rates = []
    for m in models:
        d = summary2[(summary2["model_short"] == m) & (summary2["technique"] == tech) &
                     (summary2["coefficient"] > 0)]
        r = d["brainrot_rate"].mean() if not d.empty else 0
        rates.append(r)
    ax.bar(x + (i - 1) * width, rates, width, label=TECHNIQUE_LABEL[tech], color=colors[tech])

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylabel("Mean Brainrot Rate (positive coeffs)")
ax.set_title("Model × Technique Effectiveness — Positive Steering Only (clean)", fontsize=13, fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

fig.tight_layout()
fig.savefig(OUT_DIR / "model_technique_summary.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved model_technique_summary.png")

# ── Plot 6: Steering vector norm analysis ─────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, model_full in enumerate(MODEL_SHORT):
    model_short = MODEL_SHORT[model_full]
    ax = axes[idx]
    subset = df_clean[df_clean["model_name"] == model_full].drop_duplicates(
        subset=["technique", "layer_pct", "steering_vector_norm"])[
        ["technique", "layer_pct", "steering_vector_norm"]
    ]

    for tech, marker in zip(["dom", "pca", "aas"], ["o", "s", "D"]):
        d = subset[subset["technique"] == tech].sort_values("layer_pct")
        ax.plot(d["layer_pct"], d["steering_vector_norm"], marker=marker, label=TECHNIQUE_LABEL[tech], linewidth=1.5)

    ax.set_title(model_short, fontsize=12, fontweight="bold")
    ax.set_xlabel("Layer %")
    if idx == 0:
        ax.set_ylabel("Natural style-shift norm")
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.suptitle("Natural Style-Shift Norm by Layer %", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "vector_norms.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved vector_norms.png")

# ── Plot 7: Degeneracy rate by model × coefficient ────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, model_full in enumerate(MODEL_SHORT):
    model_short = MODEL_SHORT[model_full]
    ax = axes[idx]
    subset = df_all[df_all["model_name"] == model_full]

    degen_data = []
    for t in ["dom", "pca", "aas"]:
        for coeff in sorted(subset["coefficient"].unique()):
            d = subset[(subset["technique"] == t) & (subset["coefficient"] == coeff)]
            if len(d) > 0:
                degen_rate = d["degenerate"].sum() / len(d)
                degen_data.append({"technique": t, "coefficient": coeff, "degenerate_rate": degen_rate})

    degen_df = pd.DataFrame(degen_data)
    for t, marker in zip(["dom", "pca", "aas"], ["o", "s", "D"]):
        d = degen_df[degen_df["technique"] == t].sort_values("coefficient")
        ax.plot(d["coefficient"], d["degenerate_rate"] * 100, marker=marker,
                label=TECHNIQUE_LABEL[t], linewidth=1.5)

    ax.set_title(model_short, fontsize=12, fontweight="bold")
    ax.set_xlabel("Coefficient")
    if idx == 0:
        ax.set_ylabel("Degeneracy Rate (%)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle("Response Degeneracy Rate by Coefficient (higher = more broken output)", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "degeneracy.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved degeneracy.png")

print(f"\nAll plots saved to {OUT_DIR.resolve()}/")
