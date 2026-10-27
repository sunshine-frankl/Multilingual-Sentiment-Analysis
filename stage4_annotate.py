"""Stage 4 - Annotation sheets + inter-annotator agreement (Fleiss' kappa).

  python stage4_annotate.py make --n 600          # sample target rows -> data/annotation/annotation_sheet.csv
  python stage4_annotate.py kappa FILE.csv        # compute kappa from the filled sheet
  python stage4_annotate.py gold  FILE.csv        # majority-vote gold labels -> data/annotation/gold.csv

Sheet columns to fill in: a1_polarity, a2_polarity, a3_polarity (neg/neu/pos), sarcasm (0/1), code_switch (0/1).
`cs_hint` is only a script-mixing hint (Cyrillic + Latin in one text). It is NOT a code-switching label:
Russian loanwords inside Kazakh look like switching under shared Cyrillic, so humans must decide.
"""
import argparse
import re
import sys

import numpy as np
import pandas as pd

from config import ANNOT, CLEAN, SEED

LAT, CYR = re.compile(r"[A-Za-z]"), re.compile(r"[А-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі]")
CATS = ["neg", "neu", "pos"]


def make(n):
    src = CLEAN / "clean_morph.csv"
    df = pd.read_csv(src if src.exists() else CLEAN / "clean.csv", keep_default_na=False, dtype=str)
    pool = df[df["lang"].isin(["kk", "tr", "kk-ru"]) & (df["tier"].isin(["2", "3"]))]
    per = max(n // max(pool["lang"].nunique(), 1), 1)
    s = pd.concat([g.sample(min(per, len(g)), random_state=SEED) for _, g in pool.groupby("lang")])
    s["cs_hint"] = [int(bool(LAT.search(t)) and bool(CYR.search(t))) for t in s["text"]]
    for c in ("a1_polarity", "a2_polarity", "a3_polarity", "sarcasm", "code_switch"):
        s[c] = ""
    s[["id", "text", "lang", "source", "label", "cs_hint", "a1_polarity", "a2_polarity",
       "a3_polarity", "sarcasm", "code_switch"]].to_csv(ANNOT / "annotation_sheet.csv", index=False)
    print(f"wrote {len(s)} rows -> {ANNOT/'annotation_sheet.csv'}")


def fleiss(ratings: pd.DataFrame):
    """ratings: n_items x n_raters of category strings."""
    n_items, n_r = ratings.shape
    M = np.array([[ (ratings.iloc[i] == c).sum() for c in CATS] for i in range(n_items)], float)
    P_i = ((M ** 2).sum(1) - n_r) / (n_r * (n_r - 1))
    p_j = M.sum(0) / (n_items * n_r)
    Pe = (p_j ** 2).sum()
    return (P_i.mean() - Pe) / (1 - Pe) if Pe < 1 else float("nan")


def load_filled(path):
    df = pd.read_csv(path, keep_default_na=False, dtype=str)
    cols = ["a1_polarity", "a2_polarity", "a3_polarity"]
    df = df[(df[cols] != "").all(axis=1)]
    bad = df[~df[cols].isin(CATS).all(axis=1)]
    if len(bad):
        sys.exit(f"{len(bad)} rows contain labels outside {CATS}; fix them first.")
    return df, cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "kappa", "gold"])
    ap.add_argument("file", nargs="?")
    ap.add_argument("--n", type=int, default=600)
    a = ap.parse_args()
    if a.cmd == "make":
        return make(a.n)
    df, cols = load_filled(a.file)
    if a.cmd == "kappa":
        print(f"items fully annotated: {len(df)}")
        print(f"Fleiss kappa (polarity, 3 annotators): {fleiss(df[cols]):.3f}")
        for lang, g in df.groupby("lang"):
            print(f"  {lang}: n={len(g)} kappa={fleiss(g[cols]):.3f}")
    else:
        mode = df[cols].mode(axis=1)
        df["gold"] = mode[0]
        df["tie"] = mode[1].notna() if mode.shape[1] > 1 else False
        df.to_csv(ANNOT / "gold.csv", index=False)
        print(f"gold labels: {len(df)} rows, ties (need adjudication): {int(df['tie'].sum())}")


if __name__ == "__main__":
    main()
