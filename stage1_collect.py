"""Stage 1 - Multi-source data ingestion (open datasets via Hugging Face).

Usage:
    python stage1_collect.py                 # all enabled sources
    python stage1_collect.py --only kazsandra tr_winvoker
    python stage1_collect.py --demo          # tiny synthetic data, to test the pipeline offline
Output: data/raw/<key>.csv  (unified schema, see config.py)
"""
import argparse
import itertools
import re
import sys

import pandas as pd

from config import LABEL_CANDS, RAW, SOURCES, TEXT_CANDS, norm_label

COLS = ["id", "text", "label", "label_raw", "lang", "domain", "source", "tier", "license"]


def _pick(cols, cands):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c in low:
            return low[c]
    return None


def _iter_rows(src):
    from datasets import load_dataset
    kw = {"path": src["hf_id"]}
    if src.get("data_files"):
        kw["data_files"] = src["data_files"]
    streaming = bool(src.get("streaming"))
    ds = load_dataset(streaming=streaming, **kw)
    # DatasetDict / IterableDatasetDict -> choose splits; single dataset -> wrap
    if hasattr(ds, "keys"):
        names = src["splits"] or list(ds.keys())
        parts = [ds[n] for n in names if n in ds]
    else:
        parts = [ds]
    it = itertools.chain.from_iterable(parts)
    if src.get("limit"):
        it = itertools.islice(it, src["limit"])
    return it


def _row_to_text(row, text_col):
    if "title" in row and "content" in row:        # amazon_polarity
        return f"{row['title']}. {row['content']}"
    return row.get(text_col)


def collect(src):
    rows = _iter_rows(src)
    first = next(rows)
    cols = list(first.keys())
    tcol = src.get("text_col") or _pick(cols, TEXT_CANDS) or ("content" if "content" in cols else None)
    lcol = _pick(cols, LABEL_CANDS)
    print(f"  [{src['key']}] columns={cols} -> text={tcol} label={lcol}")
    if tcol is None:
        raise ValueError(f"no text column found in {cols}; set text_col in config.py")
    excl = re.compile(src["exclude_domain_regex"], re.I) if src.get("exclude_domain_regex") else None
    out, excluded = [], 0
    for i, row in enumerate(itertools.chain([first], rows)):
        txt = _row_to_text(row, tcol)
        if not isinstance(txt, str) or not txt.strip():
            continue
        dom = str(row["dataset"]) if "dataset" in row else src["domain"]   # winvoker keeps the origin here
        if excl and excl.search(dom):
            excluded += 1
            continue
        raw = row.get(lcol) if lcol else None
        lab = norm_label(raw)
        # binary 0/1 datasets (SST-2, Yelp, Amazon): 0 = neg, 1 = pos
        if not lab and src["key"] in ("sst2", "yelp_polarity", "amazon_polarity") and raw in (0, 1):
            lab = "neg" if raw == 0 else "pos"
        out.append([f"{src['key']}_{i}", txt, lab, "" if raw is None else str(raw), src["lang"],
                    dom, src["key"], src["tier"], src["license"]])
    df = pd.DataFrame(out, columns=COLS)
    if df.empty:
        raise ValueError("0 rows collected (check text column / filters)")
    if excluded:
        print(f"  excluded {excluded:,} rows matching '{src['exclude_domain_regex']}' in the `dataset` column")
    # 1-5 star ratings in ANY label column (e.g. KazSAnDRA `label` = review score): 1-2 neg, 3 neu, 4-5 pos
    # (assumption: 3 stars = neutral; the KazSAnDRA paper's polarity task drops 3-star reviews instead)
    if lcol and (df["label"] == "").all():
        try:
            vals = {int(float(v)) for v in df["label_raw"].unique()}
        except ValueError:
            vals = set()
        if vals and vals <= {1, 2, 3, 4, 5} and vals & {4, 5}:
            v = df["label_raw"].map(lambda x: int(float(x)))
            df["label"] = v.map(lambda s: "neg" if s <= 2 else ("neu" if s == 3 else "pos"))
            print(f"  [{src['key']}] mapped 1-5 ratings -> neg/neu/pos")
    if lcol:
        n_un = int((df["label"] == "").sum())
        print(f"  label distribution: {df['label'].replace('', '(unmapped)').value_counts().to_dict()}")
        if n_un:
            print(f"  !! [{src['key']}] {n_un:,} rows have unmapped labels; raw values: "
                  f"{df.loc[df['label'] == '', 'label_raw'].unique()[:10]}. They will be ignored as labelled data. "
                  f"Check the dataset card and extend norm_label() in config.py if needed.")
    return df


def demo():
    en = ["Great movie, loved it!", "Terrible plot and bad acting.", "It was okay, nothing special.",
          "<p>Best purchase ever!!! http://x.co</p>", "Best purchase ever!!!"]
    kk = ["Өте жақсы кітап, ұсынамын", "Жаман қызмет, ұнамады", "Орташа, ештеңе айтарлық жоқ",
          "Қызмет супер, но доставка баяу болды", "Жақсылықтарыңыздан айналайын"]
    tr = ["Ürün çok kaliteli, kesinlikle tavsiye ederim", "Berbat, param çöpe gitti",
          "Fena değil ama beklentimi karşılamadı", "Kargo hızlıydı, teşekkürler", "Hiç memnun kalmadım"]
    lab = ["pos", "neg", "neu", "pos", "pos"]
    rows = []
    for lang, texts, tier in (("en", en, 1), ("kk", kk, 2), ("tr", tr, 2)):
        for rep in range(40):    # repeat with a suffix so splits have enough rows
            for t, l in zip(texts, lab):
                rows.append([f"demo_{lang}_{len(rows)}", t if rep % 8 == 0 else f"{t} #{rep}", l, l, lang,
                             "demo", "demo", tier, "synthetic"])
    pd.DataFrame(rows, columns=COLS).to_csv(RAW / "demo.csv", index=False)
    print(f"demo data written: {len(rows)} rows -> {RAW/'demo.csv'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--include-disabled", action="store_true")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo:
        return demo()
    ok, failed = [], []
    for src in SOURCES:
        if a.only and src["key"] not in a.only:
            continue
        if not src["enabled"] and not (a.include_disabled or a.only):
            print(f"skip (disabled): {src['key']} - {src['note']}")
            continue
        print(f"-> {src['key']}  ({src['hf_id']})")
        try:
            df = collect(src)
            df.to_csv(RAW / f"{src['key']}.csv", index=False)
            print(f"  saved {len(df):,} rows")
            ok.append(src["key"])
        except Exception as e:  # keep going: one broken source must not stop the run
            print(f"  FAILED: {type(e).__name__}: {str(e)[:200]}")
            if "gated" in str(e).lower() or "authenticated" in str(e).lower():
                print("  -> This dataset is gated. 1) open https://huggingface.co/datasets/" + src["hf_id"]
                      + " and accept the conditions; 2) run `hf auth login` (or export HF_TOKEN=...); 3) re-run:"
                      f" python stage1_collect.py --only {src['key']}")
            failed.append(src["key"])
    print(f"\nOK: {ok}\nFAILED: {failed}")
    sys.exit(1 if failed and not ok else 0)


if __name__ == "__main__":
    main()
