"""Audit the brainrot judge: is is_brainrot==1 really brainrot, or is it fooled
by emoji spam / broken text?

Prints ASCII-safe stats and writes judge_audit.txt (UTF-8) with categorised
example responses so a human can verify the judge's calls.
"""
from pathlib import Path
import glob, re
import numpy as np
import pandas as pd
from diagnose_steering import BRAINROT_KEYWORDS

KEY = ["model_name", "technique", "layer_pct", "coefficient", "question_id"]


def load():
    files = sorted(glob.glob("steering_results/*_dual_judged.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(subset=KEY).reset_index(drop=True)
    df["br"] = pd.to_numeric(df["is_brainrot"], errors="coerce").fillna(-1).astype(int).clip(lower=0)
    raw = df["judge_coherence_raw"].astype(str).str.upper()
    df["coh"] = np.where(raw.str.contains("INCOHERENT"), 0,
                         np.where(raw.str.contains("COHERENT"), 1, -1))
    return df


def kw_hits(t):
    tl = str(t).lower()
    return sum(1 for k in BRAINROT_KEYWORDS if k in tl)


def emoji_count(t):
    # chars in symbol/emoji ranges (rough): misc symbols, dingbats, emoji blocks
    return sum(1 for c in str(t) if ord(c) >= 0x2190)


def symbol_ratio(t):
    s = re.sub(r"\s", "", str(t))
    if not s:
        return 0.0
    keep = sum(1 for c in s if c.isalnum() or c in ".,!?'\"-:;()[]/%$&@#")
    return 1 - keep / len(s)


def degenerate(t):
    toks = str(t).split()
    if len(toks) < 5:
        return True
    if sum(toks[i] == toks[i - 1] for i in range(1, len(toks))) >= 5:
        return True
    a = re.findall(r"[a-zA-Z]", str(t))
    return bool(a) and len(set(c.lower() for c in a)) < 8


df = load()
df["kw"] = df["response"].apply(kw_hits)
df["emoji"] = df["response"].apply(emoji_count)
df["symr"] = df["response"].apply(symbol_ratio)
df["degen"] = df["response"].apply(degenerate)
df["nwords"] = df["response"].apply(lambda s: len(str(s).split()))

BR = df[df.br == 1]
CB = df[(df.br == 1) & (df.coh == 1)]   # coherent brainrot = headline metric

print("=" * 64)
print("BRAINROT JUDGE AUDIT")
print("=" * 64)
print(f"is_brainrot==1 total          : {len(BR)}")
print(f"  of which coherent (cb)      : {len(CB)}  ({len(CB)/max(1,len(BR))*100:.0f}% of brainrot)")
print(f"  of which INcoherent         : {len(BR)-len(CB)}  (excluded from headline metric)")
print()
print("Among COHERENT-BRAINROT (cb) responses:")
print(f"  with >=1 slang keyword      : {(CB.kw>=1).mean()*100:5.1f}%")
print(f"  with >=2 slang keywords     : {(CB.kw>=2).mean()*100:5.1f}%")
print(f"  with 0 slang keywords       : {(CB.kw==0).mean()*100:5.1f}%   <- scrutinise")
print(f"  with any emoji/symbol        : {(CB.emoji>=1).mean()*100:5.1f}%")
print(f"  emoji-heavy (symbol>30%)     : {(CB.symr>0.30).mean()*100:5.1f}%")
print(f"  flagged degenerate (heur.)   : {CB.degen.mean()*100:5.1f}%   <- broken text leaking in")
print(f"  median word count            : {int(CB.nwords.median())}")
print()
# Potential false positives: judge says brainrot+coherent but no slang & no emoji
FP = CB[(CB.kw == 0) & (CB.emoji == 0) & (~CB.degen)]
print(f"Potential FALSE POSITIVES (cb, 0 slang, 0 emoji, not degenerate): {len(FP)}"
      f"  ({len(FP)/max(1,len(CB))*100:.1f}% of cb)")
# Emoji-only-ish brainrot
EM = CB[(CB.kw == 0) & (CB.symr > 0.30)]
print(f"Emoji/symbol-driven brainrot (cb, 0 slang, symbol>30%)         : {len(EM)}"
      f"  ({len(EM)/max(1,len(CB))*100:.1f}% of cb)")

# ── write examples to a UTF-8 file the user can open ──
def block(title, sub, n=12):
    out = [f"\n{'='*70}\n{title}  (showing {min(n,len(sub))} of {len(sub)})\n{'='*70}"]
    for _, r in sub.sample(min(n, len(sub)), random_state=7).iterrows():
        out.append(f"\n[{r.model_name.split('/')[-1]} {r.technique} L{r.layer_pct} c={r.coefficient:+g}] "
                   f"kw={r.kw} emoji={r.emoji} sym={r.symr:.0%} coh={r.coh}")
        out.append(f"Q: {r.question}")
        out.append(f"A: {str(r.response).strip()[:600]}")
    return "\n".join(out)

doc = ["BRAINROT JUDGE AUDIT — example responses for human verification",
       "Goal: confirm is_brainrot==1 (and coherent) really IS brainrot, not emoji/broken text.\n"]
doc.append(block("(1) COHERENT-BRAINROT with slang keywords  (expected: clearly brainrot)",
                 CB[CB.kw >= 2]))
doc.append(block("(2) COHERENT-BRAINROT, NO slang keyword, NO emoji  (POSSIBLE FALSE POSITIVES)", FP))
doc.append(block("(3) COHERENT-BRAINROT, emoji-heavy, no slang  (judge call: is this brainrot?)", EM))
doc.append(block("(4) Brainrot but INCOHERENT  (correctly excluded by the coherence gate)",
                 df[(df.br == 1) & (df.coh == 0)]))
Path("judge_audit.txt").write_text("\n".join(doc), encoding="utf-8")
print("\nWrote judge_audit.txt (open it to eyeball the judge's calls).")
