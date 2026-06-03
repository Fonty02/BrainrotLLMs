"""Generate papero/sections/examples.tex with REAL prompt+response examples
covering every regime (baseline, coherent brainrot per technique/model,
overdose-degeneration, PCA, negative-steering breakage).

Real emoji are kept and rendered by the Noto Color Emoji fallback (LuaLaTeX);
only broken non-emoji glyphs (CJK fragments, replacement chars) are dropped.
"""
import glob, re
import numpy as np
import pandas as pd
from diagnose_steering import BRAINROT_KEYWORDS

KEY = ["model_name", "technique", "layer_pct", "coefficient", "question_id"]
MS = {"Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B",
      "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
      "google/gemma-4-E2B-it": "Gemma-4-2B"}
TL = {"dom": "DoM", "pca": "PCA", "aas": "AAS"}

# Unicode ranges kept as real emoji/symbols (rendered via Noto Color Emoji).
EMOJI_RANGES = [
    (0x20E3, 0x20E3), (0x2190, 0x21FF), (0x2300, 0x23FF),
    (0x2460, 0x24FF), (0x25A0, 0x27BF), (0x2900, 0x297F), (0x2B00, 0x2BFF),
    (0xFE00, 0xFE0F), (0x1F000, 0x1FAFF),
]  # ZWJ (U+200D) deliberately excluded: dangling joiners hit nullfont


def is_emoji(o):
    return any(lo <= o <= hi for lo, hi in EMOJI_RANGES)


def emoji_count(text):
    return sum(1 for c in str(text) if is_emoji(ord(c)))


def load():
    df = pd.concat([pd.read_csv(f) for f in glob.glob("steering_results/*_dual_judged.csv")],
                   ignore_index=True).drop_duplicates(subset=KEY).reset_index(drop=True)
    raw = df["judge_coherence_raw"].astype(str).str.upper()
    df["coh"] = np.where(raw.str.contains("INCOHERENT"), 0,
                         np.where(raw.str.contains("COHERENT"), 1, -1))
    df["br"] = pd.to_numeric(df["is_brainrot"], errors="coerce").fillna(0).clip(lower=0).astype(int)
    df["m"] = df["model_name"].map(MS)
    df["kw"] = df["response"].apply(lambda t: sum(1 for k in BRAINROT_KEYWORDS if k in str(t).lower()))
    return df


def sanitize(text, max_words=55):
    t = str(text)
    for a, b in [("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'),
                 ("–", "--"), ("—", "---"), ("…", "..."), (" ", " ")]:
        t = t.replace(a, b)
    # keep ASCII + real emoji; drop other non-ASCII (broken glyphs)
    t = "".join(c if (ord(c) < 128 or is_emoji(ord(c))) else " " for c in t)
    t = re.sub(r"\s+", " ", t).strip()
    words = t.split()
    if len(words) > max_words:
        t = " ".join(words[:max_words]) + " [...]"
    # LaTeX-escape ASCII specials (emoji pass through untouched as UTF-8)
    t = t.replace("\\", "\x00")  # sentinel for literal backslash
    for a, b in [("{", r"\{"), ("}", r"\}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        t = t.replace(a, b)
    t = t.replace("\x00", r"\textbackslash{}")
    return t


def readable_score(row):
    """Higher = nicer to print: rich in slang, not dominated by emoji spam."""
    return row["kw"] * 2 - emoji_count(row["response"]) * 0.3


def coeff_str(r):
    return (f"$\\theta={int(r.coefficient)}^\\circ$" if r.technique == "aas"
            else f"$c={int(r.coefficient)}$")


def pick(df, n=1, **filt):
    sub = df.copy()
    for k, v in filt.items():
        if k == "coeff":
            sub = sub[sub.coefficient == v]
        elif k == "coeff_in":
            sub = sub[sub.coefficient.isin(v)]
        else:
            sub = sub[sub[k] == v]
    if sub.empty:
        return []
    sub = sub.assign(_s=sub.apply(readable_score, axis=1)).sort_values("_s", ascending=False)
    return [sub.iloc[i] for i in range(min(n, len(sub)))]


df = load()
out = []
def w(s=""): out.append(s)


def block(r, tag):
    hdr = f"{r.m} $\\cdot$ {TL[r.technique]} $\\cdot$ L{int(r.layer_pct)}\\% $\\cdot$ {coeff_str(r)}"
    w(r"\medskip\noindent\textbf{" + hdr + r"}\hfill\textit{(" + tag + r")}\\")
    w(r"\emph{Prompt:} " + sanitize(r["question"], 40) + r"\\")
    w(r"{\itshape\small \emph{Response:} " + sanitize(r["response"]) + r"}")
    w(r"\par\vspace{3pt}\hrule")


w(r"\section{Qualitative Examples}")
w(r"\label{sec:examples}")
w("")
w(r"""This appendix shows representative, verbatim prompt--response pairs for each
regime of the study (model judgements in parentheses). Emoji are rendered as in
the original output; broken non-emoji glyphs from degenerate generations are
dropped, and long responses are truncated with \texttt{[...]}.""")
w("")

w(r"\subsection{Baseline: weak steering leaves the model unchanged}")
for r in pick(df, 1, technique="dom", coeff=1.0, coh=1, br=0):
    block(r, "coherent, not brainrot")

w(r"\subsection{Coherent brainrot (the success regime)}")
w(r"\paragraph{Difference of Means.}")
for m in ["Llama-3.1-8B", "Qwen2.5-7B", "Gemma-4-2B"]:
    for r in pick(df[(df.m == m) & (df.kw >= 2)], 1, technique="dom", coh=1, br=1, coeff_in=[2.0, 3.0]):
        block(r, "coherent brainrot")
w(r"\paragraph{Angular Activation Steering.}")
for m in ["Llama-3.1-8B", "Qwen2.5-7B"]:
    for r in pick(df[(df.m == m) & (df.kw >= 2)], 1, technique="aas", coh=1, br=1, coeff=35.0):
        block(r, "coherent brainrot")

w(r"\subsection{Overdose: strong steering breaks coherence}")
for r in pick(df, 1, technique="dom", coeff=4.0, coh=0, br=1):
    block(r, "brainrot but INCOHERENT")
for r in pick(df, 1, technique="aas", coeff=45.0, coh=0, br=1):
    block(r, "brainrot but INCOHERENT")

w(r"\subsection{PCA: the direction that does not transfer}")
for r in pick(df, 1, technique="pca", coeff_in=[3.0, 4.0]):
    block(r, "PCA, weak style effect")

w(r"\subsection{Negative steering: magnitude breaks, sign gives no style}")
for r in pick(df, 1, technique="dom", coeff=-4.0, coh=0):
    block(r, "anti-brainrot direction, incoherent")

text = "\n".join(out)
open("papero/sections/examples.tex", "w", encoding="utf-8").write(text)
print("wrote papero/sections/examples.tex")
print("blocks:", text.count(r"\hrule"))
