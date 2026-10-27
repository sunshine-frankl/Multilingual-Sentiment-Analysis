"""Stage 5 - Dataset splitting for Zero-Shot vs Joint fine-tuning.

Outputs in data/splits/:
  source_train / source_dev / source_test   EN labelled data, 80/10/10 (stratified by label)
  target_train / target_dev / target_test   KK+TR labelled data, 60/10/30 (stratified by lang+label)
  joint_train                               balanced 50/50 sample of source_train and target_train
  unlabelled_target                         Tier-4 / unlabelled KK text for domain adaptation
  cs_probe                                  code-switched probe set (no labels)
  split_stats.csv                           counts per split / lang / label

Design note: zero-shot is trained on source_train ONLY and evaluated on target_test. Joint training uses
joint_train and is evaluated on the SAME target_test, so the two paradigms are directly comparable and the
target data used for training never leaks into evaluation.
"""
import argparse

import pandas as pd

from config import CLEAN, SEED, SPLITS


def strat_split(df, fracs, by):
    parts = {k: [] for k in fracs}
    for _, g in df.groupby(by):
        g = g.sample(frac=1, random_state=SEED)
        n, i = len(g), 0
        keys = list(fracs)
        for j, k in enumerate(keys):
            m = n - i if j == len(keys) - 1 else int(round(n * fracs[k]))
            parts[k].append(g.iloc[i:i + m]); i += m
    return {k: pd.concat(v) for k, v in parts.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", action="store_true", help="drop neutral (EN sources are binary)")
    a = ap.parse_args()
    f = CLEAN / "clean_morph.csv"
    df = pd.read_csv(f if f.exists() else CLEAN / "clean.csv", keep_default_na=False, dtype=str)

    lab = df[df["label"].isin(["neg", "neu", "pos"])]
    if a.binary:
        lab = lab[lab["label"] != "neu"]
    out = {}
    en = lab[lab["lang"] == "en"]
    tgt = lab[lab["lang"].isin(["kk", "tr"])]
    if len(en):
        s = strat_split(en, {"train": .8, "dev": .1, "test": .1}, ["label"])
        out.update({f"source_{k}": v for k, v in s.items()})
    if len(tgt):
        t = strat_split(tgt, {"train": .6, "dev": .1, "test": .3}, ["lang", "label"])
        out.update({f"target_{k}": v for k, v in t.items()})
    if len(en) and len(tgt):
        n = min(len(out["source_train"]), len(out["target_train"]))
        out["joint_train"] = pd.concat([out["source_train"].sample(n, random_state=SEED),
                                        out["target_train"].sample(n, random_state=SEED)]
                                       ).sample(frac=1, random_state=SEED)
    out["unlabelled_target"] = df[(df["lang"] == "kk") & (df["label"] == "")]
    out["cs_probe"] = df[df["lang"] == "kk-ru"]

    # leakage guard: no text shared between train and any eval split
    def keys(x): return set(x["text"].str.lower())
    for tr, ev in (("source_train", "source_test"), ("target_train", "target_test"), ("joint_train", "target_test")):
        if tr in out and ev in out:
            assert not (keys(out[tr]) & keys(out[ev])), f"LEAK between {tr} and {ev}"

    rows = []
    for name, d in out.items():
        d.to_csv(SPLITS / f"{name}.csv", index=False)
        for (lang, l), c in d.groupby(["lang", "label"]).size().items():
            rows.append([name, lang, l or "(none)", c])
        if d.empty:
            rows.append([name, "-", "-", 0])
    st = pd.DataFrame(rows, columns=["split", "lang", "label", "n"])
    st.to_csv(SPLITS / "split_stats.csv", index=False)
    print(st.pivot_table(index="split", columns="label", values="n", aggfunc="sum", fill_value=0).to_string())


if __name__ == "__main__":
    main()
