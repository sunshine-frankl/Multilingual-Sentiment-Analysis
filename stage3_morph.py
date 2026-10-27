"""Stage 3 - Morphology-aware filter: Root-Preserving Subword Constraint (RPSC).

For every whitespace word the script counts sub-tokens produced by a multilingual tokenizer and flags
words with > MAX_SUB sub-tokens (over-segmentation, default 4 as in the DCP paper).
Rows are TAGGED by default (rpsc_flag); use --drop to remove flagged rows from the training corpus.
It also writes per-language "fertility" statistics (sub-tokens per word) - a ready-made table for the paper.

Tokenizer: bert-base-multilingual-cased (WordPiece, matches the '##' example in the paper) or
           xlm-roberta-base (SentencePiece). If the tokenizer cannot be downloaded the script stops,
           unless --heuristic is given (rough length-based estimate; NOT for reported results).
"""
import argparse
import re

import pandas as pd

from config import CLEAN

WORD = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*", re.UNICODE)


def load_tok(name, heuristic):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(name, use_fast=True)
    except Exception as e:
        if heuristic:
            print(f"!! tokenizer unavailable ({type(e).__name__}); using length heuristic (approximation)")
            return None
        raise SystemExit(f"Cannot load tokenizer '{name}': {e}\nRe-run with internet or --heuristic.")


def n_sub(tok, words):
    if tok is None:
        return [max(1, -(-len(w) // 4)) for w in words]       # ceil(len/4)
    enc = tok(words, add_special_tokens=False)["input_ids"]
    return [len(x) for x in enc]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="bert-base-multilingual-cased")
    ap.add_argument("--max-sub", type=int, default=4)
    ap.add_argument("--drop", action="store_true")
    ap.add_argument("--heuristic", action="store_true")
    a = ap.parse_args()

    df = pd.read_csv(CLEAN / "clean.csv", keep_default_na=False, dtype=str)
    tok = load_tok(a.tokenizer, a.heuristic)

    # tokenise the vocabulary of words once (much faster than per-row)
    all_words = sorted({w for t in df["text"] for w in WORD.findall(t)})
    cache = dict(zip(all_words, n_sub(tok, all_words))) if all_words else {}

    mx, over, fert = [], [], []
    for t in df["text"]:
        ws = WORD.findall(t)
        c = [cache[w] for w in ws] or [0]
        mx.append(max(c)); over.append(sum(x > a.max_sub for x in c))
        fert.append(sum(c) / max(len(ws), 1))
    df["max_subtokens"], df["n_overseg_words"], df["fertility"] = mx, over, [round(f, 3) for f in fert]
    df["rpsc_flag"] = (df["n_overseg_words"] > 0).astype(int)

    stats = (df.assign(fertility=df["fertility"].astype(float), rpsc_flag=df["rpsc_flag"].astype(int))
               .groupby("lang").agg(rows=("text", "size"), mean_fertility=("fertility", "mean"),
                                    pct_rows_flagged=("rpsc_flag", "mean")).round(3))
    stats["pct_rows_flagged"] *= 100
    stats.to_csv(CLEAN / "morph_stats.csv")
    print(stats.to_string())

    if a.drop:
        df = df[df["rpsc_flag"] == 0]
        print(f"--drop: kept {len(df):,} rows")
    df.to_csv(CLEAN / "clean_morph.csv", index=False)


if __name__ == "__main__":
    main()
