"""Analyze steering experiment results and produce a summary .txt report.

Degenerate (garbled/repetitive) responses are filtered out before computing
brainrot rates — the judge alone cannot distinguish real brainrot from broken text.
"""

from pathlib import Path
import re
import pandas as pd
import numpy as np

RESULTS_DIR = Path("steering_results")
OUT = Path("analysis_report.txt")

MODEL_SHORT = {
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "google/gemma-4-E2B-it": "Gemma-4-2B",
}
TECH_LABEL = {"dom": "DoM", "pca": "PCA", "aas": "AAS"}


# ── Degeneracy filter ────────────────────────────────────────────
def is_degenerate(text: str) -> bool:
    """Return True if the response is garbled/broken/non-text."""
    t = str(text).strip()
    if len(t) < 20:
        return True  # too short to be a real answer
    tokens = t.split()
    if len(tokens) < 5:
        return True
    # Consecutive repetition of same token (e.g. "I, I, I, I, I...")
    runs = 0
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i - 1]:
            runs += 1
    if runs >= 5:
        return True
    # Very few unique alpha characters = gibberish
    alpha_chars = re.findall(r"[a-zA-Z]", t)
    if len(alpha_chars) > 0:
        unique_count = len(set(c.lower() for c in alpha_chars))
        if unique_count < 8:
            return True
    return False


# ── Load ─────────────────────────────────────────────────────────
# Prefer dual-judged files (brainrot + coherence). Fall back to brainrot-only.
dual_files = sorted(RESULTS_DIR.glob("*_dual_judged.csv"))
if dual_files:
    files = dual_files
else:
    files = [f for f in sorted(RESULTS_DIR.glob("*_judged.csv"))
             if not f.name.endswith("_dual_judged.csv")]

dfs = [pd.read_csv(f) for f in files]
df = pd.concat(dfs, ignore_index=True)
df["model_short"] = df["model_name"].map(MODEL_SHORT)
df["coeff_sign"] = np.where(df["coefficient"] > 0, "positive", "negative")
df["is_brainrot"] = pd.to_numeric(df["is_brainrot"], errors="coerce").fillna(-1).astype(int)

# Degeneracy / breakage gate. If the coherence judge was run, use it (a response
# only counts as real brainrot if it is BOTH brainrot AND coherent); otherwise
# fall back to the text heuristic.
if "is_coherent" in df.columns:
    df["is_coherent"] = pd.to_numeric(df["is_coherent"], errors="coerce").fillna(-1).astype(int)
    df["degenerate"] = df["is_coherent"] != 1
    GATE = "judge coherence (is_coherent != 1)"
else:
    df["degenerate"] = df["response"].apply(is_degenerate)
    GATE = "text heuristic"

total = len(df)
degen_count = int(df["degenerate"].sum())
clean = df[~df["degenerate"]]
print(f"Degeneracy gate: {GATE}")
print(f"Filtered: {degen_count}/{total} degenerate responses removed ({degen_count/total*100:.1f}%)")
print(f"Remaining clean responses: {len(clean)}")

categories = sorted(clean["question_category"].dropna().unique())


def br_rate(subset):
    n = len(subset)
    if n == 0:
        return 0, 0, 0, 0
    k = (subset["is_brainrot"] == 1).sum()
    r = k / n
    se = np.sqrt(r * (1 - r) / n)
    lo = max(0, r - 1.96 * se)
    hi = min(1, r + 1.96 * se)
    return r, lo, hi, n


def fmt_r(r, lo, hi):
    return f"{r*100:.1f}% [{lo*100:.1f}–{hi*100:.1f}%]"


lines = []
def w(s=""):
    lines.append(s)


# ══════════════════════════════════════════════════════════════════
w("╔══════════════════════════════════════════════════════════════╗")
w("║   BRAINROT ACTIVATION STEERING — ANALYSIS REPORT           ║")
w("║   (degenerate responses filtered out)                      ║")
w("╚══════════════════════════════════════════════════════════════╝")
w()
w(f"Generated from {len(dfs)} judged CSV files ({total} responses total)")
w(f"Degeneracy/breakage gate: {GATE}")
w(f"Degenerate responses removed: {degen_count} ({degen_count/total*100:.1f}%)")
w(f"Clean responses analyzed: {len(clean)}")
if "is_coherent" in df.columns:
    w("NOTE: brainrot rates below are 'coherent brainrot' = brainrot AND coherent.")
w(f"Models: {', '.join(MODEL_SHORT.values())}")
w(f"Techniques: DoM, PCA, AAS")
w(f"Coefficients: -25, -10, -5, +5, +10, +25")
w(f"Layer %: 25%, 50%, 75%")
w(f"Categories: {', '.join(categories)}")
w()

# ══════════════════════════════════════════════════════════════════
w("=" * 60)
w("1. OVERALL BRAINROT RATES BY MODEL (clean only)")
w("=" * 60)
w()
for m_full, m_short in MODEL_SHORT.items():
    r, lo, hi, n = br_rate(clean[clean["model_name"] == m_full])
    w(f"  {m_short:18s}  {fmt_r(r, lo, hi)}  (n={n})")
w()

# ══════════════════════════════════════════════════════════════════
w("=" * 60)
w("2. DOSE-RESPONSE: MOST BRAINROTTED CONFIGURATIONS (Top 10)")
w("=" * 60)
w()

configs = []
for (m, t, lp, c), g in clean.groupby(["model_short", "technique", "layer_pct", "coefficient"]):
    r, lo, hi, n = br_rate(g)
    configs.append((m, t, lp, c, r, lo, hi, n))
configs.sort(key=lambda x: -x[4])

w(f"  {'Rank':<5} {'Model':<15} {'Technique':<6} {'Layer':<6} {'Coeff':>6} {'Rate':>22} {'N':>5}")
w(f"  {'─'*4:<5} {'─'*14:<15} {'─'*5:<6} {'─'*5:<6} {'─'*5:>6} {'─'*21:>22} {'─'*4:>5}")

for rank, (m, t, lp, c, r, lo, hi, n) in enumerate(configs[:10], 1):
    w(f"  {rank:<5} {m:<15} {TECH_LABEL[t]:<6} L{lp}%   {c:>+5.0f}  {fmt_r(r, lo, hi):>22} {n:>5}")

w()
w("  Worst 3 (least brainrot):")
for m, t, lp, c, r, lo, hi, n in configs[-3:]:
    w(f"    {m:<15} {TECH_LABEL[t]:<6} L{lp}% {c:>+5.0f}  {fmt_r(r, lo, hi)} (n={n})")
w()

# ══════════════════════════════════════════════════════════════════
w("=" * 60)
w("3. TECHNIQUE COMPARISON (pooled across models, positive coeffs only)")
w("=" * 60)
w()

pos = clean[clean["coeff_sign"] == "positive"]
for t in ["dom", "pca", "aas"]:
    r, lo, hi, n = br_rate(pos[pos["technique"] == t])
    w(f"  {TECH_LABEL[t]:6s}  {fmt_r(r, lo, hi)}  (n={n})")
w()

w()
w("  Technique delta by model (positive coeffs only):")
w()
for m_full, m_short in MODEL_SHORT.items():
    mp = pos[pos["model_name"] == m_full]
    rates = {}
    for t in ["dom", "pca", "aas"]:
        rates[t], _, _, _ = br_rate(mp[mp["technique"] == t])
    all_zero = all(v == 0 for v in rates.values())
    if all_zero:
        w(f"  {m_short:15s}  All techniques: 0.0%")
    else:
        best = max(rates, key=rates.get)
        worst = min(rates, key=rates.get)
        w(f"  {m_short:15s}  Best: {TECH_LABEL[best]} ({rates[best]*100:.1f}%)  "
          f"Worst: {TECH_LABEL[worst]} ({rates[worst]*100:.1f}%)  "
          f"Δ = {(rates[best]-rates[worst])*100:.1f}pp")
w()

# ══════════════════════════════════════════════════════════════════
w("=" * 60)
w("4. LAYER % COMPARISON (pooled, positive coeffs only)")
w("=" * 60)
w()
for lp in [25, 50, 75]:
    r, lo, hi, n = br_rate(pos[pos["layer_pct"] == lp])
    w(f"  L{lp}%   {fmt_r(r, lo, hi)}  (n={n})")
w()

# ══════════════════════════════════════════════════════════════════
w("=" * 60)
w("5. POSITIVE vs NEGATIVE COEFFICIENTS (per model)")
w("=" * 60)
w()
for m_full, m_short in MODEL_SHORT.items():
    mdf = clean[clean["model_name"] == m_full]
    r_pos, lo_pos, hi_pos, _ = br_rate(mdf[mdf["coeff_sign"] == "positive"])
    r_neg, lo_neg, hi_neg, _ = br_rate(mdf[mdf["coeff_sign"] == "negative"])
    delta = r_pos - r_neg
    w(f"  {m_short:15s}  Pos: {fmt_r(r_pos, lo_pos, hi_pos)}  "
      f"Neg: {fmt_r(r_neg, lo_neg, hi_neg)}  "
      f"Δ = {delta*100:+.1f}pp")
w()

# ══════════════════════════════════════════════════════════════════
w("=" * 60)
w(f"6. DEGENERACY/BREAKAGE REPORT — how much does steering break each model?")
w(f"   (gate: {GATE})")
w("=" * 60)
w()
for m_full, m_short in MODEL_SHORT.items():
    mdf = df[df["model_name"] == m_full]
    total_m = len(mdf)
    degen_m = mdf["degenerate"].sum()
    w(f"  {m_short:15s}  {degen_m}/{total_m} degenerate ({degen_m/total_m*100:.1f}%)")

    # Per-coefficient breakdown
    w(f"    By coefficient:")
    for coeff in sorted(mdf["coefficient"].unique()):
        mdc = mdf[mdf["coefficient"] == coeff]
        d = mdc["degenerate"].sum()
        w(f"      coeff={coeff:>+5.0f}: {d}/{len(mdc)} degenerate ({d/len(mdc)*100:.1f}%)")
    w()

# ══════════════════════════════════════════════════════════════════
w("=" * 60)
w("7. CATEGORY BREAKDOWN (per model, positive coeffs only)")
w("=" * 60)
w()
for m_full, m_short in MODEL_SHORT.items():
    w(f"  {m_short}:")
    mdf = pos[pos["model_name"] == m_full]
    for cat in categories:
        r, lo, hi, n = br_rate(mdf[mdf["question_category"] == cat])
        w(f"    {cat:<24s}  {fmt_r(r, lo, hi)}  (n={n})")
    w()

# ══════════════════════════════════════════════════════════════════
w("=" * 60)
w("8. EXAMPLE RESPONSES (true brainrot, not degenerate)")
w("=" * 60)
w()

for m_full, m_short in MODEL_SHORT.items():
    examples = clean[(clean["model_name"] == m_full) & (clean["is_brainrot"] == 1) &
                     (clean["coefficient"] >= 10)]
    if len(examples) == 0:
        examples = clean[(clean["model_name"] == m_full) & (clean["is_brainrot"] == 1)]
    if len(examples) == 0:
        w(f"  ── {m_short}: no genuine brainrot examples found ──")
        w()
        continue
    examples = examples.sample(n=min(3, len(examples)), random_state=42)

    w(f"  ── {m_short} ──")
    for _, ex in examples.iterrows():
        resp = str(ex["response"]).replace("\n", " ").strip()[:400]
        w(f"  [{TECH_LABEL[ex['technique']]} L{ex['layer_pct']}% coeff={ex['coefficient']:+.0f}]")
        w(f"  Q: {ex['question']}")
        w(f"  A: {resp}...")
        w()
w()

# ══════════════════════════════════════════════════════════════════
w("=" * 60)
w("9. DEGENERATE RESPONSE EXAMPLES (what was filtered out)")
w("=" * 60)
w()

for m_full, m_short in MODEL_SHORT.items():
    degen_ex = df[(df["model_name"] == m_full) & (df["degenerate"])]
    if len(degen_ex) == 0:
        w(f"  ── {m_short}: no degenerate responses ──")
        continue

    # Show high-coeff degenerates only (most interesting)
    degen_high = degen_ex[degen_ex["coefficient"].abs() >= 10]
    if len(degen_high) == 0:
        degen_high = degen_ex
    degen_high = degen_high.sample(n=min(3, len(degen_high)), random_state=42)

    w(f"  ── {m_short} ({len(degen_ex)} degenerate total) ──")
    for _, ex in degen_high.iterrows():
        resp = str(ex["response"]).replace("\n", " ").strip()[:200]
        was_judged_br = "YES" if ex["is_brainrot"] == 1 else "NO"
        w(f"  [{TECH_LABEL[ex['technique']]} L{ex['layer_pct']}% coeff={ex['coefficient']:+.0f}] "
          f"(judge said: {was_judged_br})")
        w(f"  Q: {ex['question']}")
        w(f"  A: {resp}...")
        w()
w()

# ══════════════════════════════════════════════════════════════════
w("=" * 60)
w("10. DOES NEGATIVE STEERING REDUCE BRAINROT BELOW BASELINE?")
w("=" * 60)
w()

for m_full, m_short in MODEL_SHORT.items():
    mdf = clean[clean["model_name"] == m_full]
    for t in ["dom", "pca", "aas"]:
        mdt = mdf[mdf["technique"] == t]
        r_pos, _, _, _ = br_rate(mdt[mdt["coefficient"].isin([25])])
        r_neg, _, _, _ = br_rate(mdt[mdt["coefficient"].isin([-25])])
        r_weak_neg, _, _, _ = br_rate(mdt[mdt["coefficient"].isin([-5])])
        w(f"  {m_short:15s} {TECH_LABEL[t]:6s}  coeff=+25: {r_pos*100:.1f}%  "
          f"coeff=-25: {r_neg*100:.1f}%  coeff=-5: {r_weak_neg*100:.1f}%")
w()

# ══════════════════════════════════════════════════════════════════
w("=" * 60)
w("11. KEY TAKEAWAYS")
w("=" * 60)
w()

model_rates = {}
for m in MODEL_SHORT.values():
    r, _, _, _ = br_rate(pos[pos["model_short"] == m])
    model_rates[m] = r
best_model = max(model_rates, key=model_rates.get)
worst_model = min(model_rates, key=model_rates.get)

tech_rates = {}
for t in ["dom", "pca", "aas"]:
    r, _, _, _ = br_rate(pos[pos["technique"] == t])
    tech_rates[t] = r
best_tech = max(tech_rates, key=tech_rates.get)

layer_rates = {}
for lp in [25, 50, 75]:
    r, _, _, _ = br_rate(pos[pos["layer_pct"] == lp])
    layer_rates[lp] = r
best_layer = max(layer_rates, key=layer_rates.get)

if configs:
    top_m, top_t, top_lp, top_c, top_r, top_lo, top_hi, _ = configs[0]
else:
    top_m, top_t, top_lp, top_c, top_r = "-", "-", "-", "-", 0

neg_effectiveness = {}
for m_full, m_short in MODEL_SHORT.items():
    mdf = clean[clean["model_name"] == m_full]
    r_pos_all, _, _, _ = br_rate(mdf[mdf["coeff_sign"] == "positive"])
    r_neg_all, _, _, _ = br_rate(mdf[mdf["coeff_sign"] == "negative"])
    neg_effectiveness[m_short] = r_pos_all - r_neg_all
best_neg_model = max(neg_effectiveness, key=neg_effectiveness.get)

# Degeneracy stats
degen_by_model = {}
for m in MODEL_SHORT.values():
    mdf = df[df["model_short"] == m]
    degen_by_model[m] = mdf["degenerate"].sum() / len(mdf) * 100

w(f"  • Most brainrottable model (genuine): {best_model} ({model_rates[best_model]*100:.1f}%)")
w(f"  • Least brainrottable model: {worst_model} ({model_rates[worst_model]*100:.1f}%)")
w(f"  • Most effective technique: {TECH_LABEL[best_tech]} ({tech_rates[best_tech]*100:.1f}%)")
w(f"  • Best layer: L{best_layer}% ({layer_rates[best_layer]*100:.1f}%)")
w(f"  • Single best config: {top_m} × {TECH_LABEL[top_t]} × L{top_lp}% × coeff={top_c:+.0f} "
    f"→ {top_r*100:.1f}% brainrot rate")
w(f"  • Largest pos↔neg gap: {best_neg_model} "
    f"({neg_effectiveness[best_neg_model]*100:.1f}pp)")
w()
w(f"  • Qwen2.5-7B degenerate rate: {degen_by_model.get('Qwen2.5-7B', 0):.1f}%")
w(f"  • Llama-3.1-8B degenerate rate: {degen_by_model.get('Llama-3.1-8B', 0):.1f}%")
w(f"  • Gemma-4-2B degenerate rate: {degen_by_model.get('Gemma-4-2B', 0):.1f}%")
w()
w("  Interpretation:")
max_genuine = max(model_rates.values()) if model_rates else 0
if max_genuine > 0.3:
    w(f"    ✅ Steering WORKS — genuine brainrot induced at {max_genuine*100:.1f}% rate")
elif max_genuine > 0.05:
    w(f"    ⚠️ Steering is MODERATE — best genuine brainrot rate is {max_genuine*100:.1f}%")
else:
    w(f"    ❌ Steering FAILS — no model produces genuine brainrot above 5%")
w()

w("=" * 60)
w("END OF REPORT")
w("=" * 60)

# ── Write ────────────────────────────────────────────────────────
OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"Report written to {OUT.resolve()} ({len(lines)} lines)")
