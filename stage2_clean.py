"""Stage 2 - Quality filtering & deduplication.

Removes: HTML artefacts, URLs/mentions (normalised to tokens), too-short texts, bot-like / spammy text,
wrong-script text, exact duplicates and near-duplicates (MinHash-LSH, Jaccard >= 0.85 on word 3-shingles).
Input : data/raw/*.csv     Output: data/clean/clean.csv + data/clean/clean_report.csv
"""
import argparse
import html
import re
import unicodedata

import pandas as pd

from config import CLEAN, RAW

TAG = re.compile(r"<[^>]+>")
URL = re.compile(r"https?://\S+|www\.\S+")
MENTION = re.compile(r"@\w+")
WS = re.compile(r"\s+")
LAT = re.compile(r"[A-Za-zÇĞİÖŞÜçğıöşü]")
CYR = re.compile(r"[А-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі]")


def normalise(t: str) -> str:
    t = html.unescape(str(t))
    t = TAG.sub(" ", t)
    t = URL.sub("<URL>", t)
    t = MENTION.sub("@user", t)
    t = unicodedata.normalize("NFKC", t)
    return WS.sub(" ", t).strip()


def is_bot_like(t: str) -> bool:
    words = t.split()
    if not words:
        return True
    alpha = sum(c.isalpha() for c in t) / max(len(t), 1)
    top = max((words.count(w) for w in set(words)), default=0) / len(words)
    return alpha < 0.4 or (len(words) >= 6 and top > 0.5) or t.count("<URL>") >= 3


def wrong_script(t: str, lang: str) -> bool:
    lat, cyr = len(LAT.findall(t)), len(CYR.findall(t))
    if lang in ("en", "tr"):
        return cyr > lat                      # mostly Cyrillic in an EN/TR row
    if lang == "kk":
        return (lat + cyr) == 0               # no letters from either script
    return False                              # kk-ru / others: keep


def near_dedup(df: pd.DataFrame, threshold=0.85):
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        print("  datasketch missing -> near-duplicate step skipped (pip install datasketch)")
        return df, 0
    keep, dropped = [], 0
    for lang, g in df.groupby("lang"):
        lsh = MinHashLSH(threshold=threshold, num_perm=64)
        for idx, text in zip(g.index, g["text"].str.lower()):
            w = text.split()
            sh = {" ".join(w[i:i + 3]) for i in range(max(len(w) - 2, 1))}
            m = MinHash(num_perm=64)
            for s in sh:
                m.update(s.encode("utf8"))
            if lsh.query(m):
                dropped += 1
                continue
            lsh.insert(str(idx), m)
            keep.append(idx)
    return df.loc[sorted(keep)], dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-words", type=int, default=3)
    ap.add_argument("--no-near-dedup", action="store_true")
    a = ap.parse_args()

    files = sorted(RAW.glob("*.csv"))
    if not files:
        raise SystemExit("No raw files. Run stage1_collect.py first.")
    df = pd.concat([pd.read_csv(f, keep_default_na=False, dtype=str) for f in files], ignore_index=True)
    rep = [("input", len(df))]
    raw_counts = df.groupby("source").size()

    df["text"] = df["text"].map(normalise)
    df = df[df["text"].str.split().str.len() >= a.min_words]; rep.append(("min_words", len(df)))
    df = df[~df["text"].map(is_bot_like)]; rep.append(("bot_like_removed", len(df)))
    df = df[[not wrong_script(t, l) for t, l in zip(df["text"], df["lang"])]]; rep.append(("script_ok", len(df)))

    # NOTE: do NOT use pandas .str.replace(r"\W+") here: with Arrow-backed strings (pandas 3) \W is ASCII-only,
    # which turns every Cyrillic text into an empty key and wipes out Kazakh data as "duplicates".
    key = df["text"].map(lambda t: re.sub(r"\W+", " ", t.lower()).strip())
    df = df[~key.duplicated()]; rep.append(("exact_dedup", len(df)))
    if not a.no_near_dedup:
        df, n = near_dedup(df.reset_index(drop=True)); rep.append(("near_dedup", len(df)))

    df.to_csv(CLEAN / "clean.csv", index=False)
    pd.DataFrame(rep, columns=["step", "rows_after"]).to_csv(CLEAN / "clean_report.csv", index=False)
    print(pd.DataFrame(rep, columns=["step", "rows_after"]).to_string(index=False))
    print(df.groupby(["lang", "source"]).size().to_string())
    # safeguard: a source that loses most of its rows almost always means a bug, not real duplicates
    kept = df.groupby("source").size().reindex(raw_counts.index, fill_value=0)
    ret = (kept / raw_counts).round(3)
    print("\nretention per source (clean / raw):\n" + ret.to_string())
    for s_, r_ in ret.items():
        if r_ < 0.5:
            print(f"!! WARNING: '{s_}' kept only {r_:.0%} of its rows. Check for a cleaning bug or a very repetitive source.")


if __name__ == "__main__":
    main()
