# DCP: Data Collection Process (CSS 410, Multilingual Sentiment)

Run on a machine with internet access (Hugging Face must be reachable):

```
pip install -r requirements.txt
python stage1_collect.py          # Stage 1: download open datasets -> data/raw/
python stage2_clean.py            # Stage 2: cleaning + exact/near dedup -> data/clean/clean.csv
python stage3_morph.py            # Stage 3: RPSC filter, fertility stats (bert-base-multilingual-cased)
python stage5_split.py            # splits for zero-shot vs joint -> data/splits/
python stage4_annotate.py make --n 600   # annotation sheet for 3 annotators
python stage4_annotate.py kappa data/annotation/annotation_sheet.csv   # after annotation
```
Offline smoke test with synthetic data: `python run_all.py --demo` (the `--demo` data is fake, never report it).

## Sources (verified by web search, 28 Oct 2026; re-check licences on each card)
| Tier | Source | Lang | Notes |
|---|---|---|---|
| 1 | stanfordnlp/sst2, fancyzhx/yelp_polarity, fancyzhx/amazon_polarity | EN | binary labels |
| 2 | issai/kazsandra (CC BY 4.0) | KK | 180,064 reviews, scores 1-5 |
| 2 | winvoker/turkish-sentiment-analysis-dataset, maydogan/Turkish_SentimentAnalysis_TRSAv1 | TR | 3-class reviews/tweets; TRSAv1 150k |
| 3 | BorisovMaksim/kk_ru_csw (KRCS) | KK-RU | 618 code-switched sentences, no sentiment labels |
| 4 | kz-transformers/multidomain-kazakh-dataset (cc100, oscar, news, books, leipzig) | KK | unlabelled, multi-GB |
Manual / on request: see `MANUAL_SOURCES` in config.py (SarcasTurk, multilingual irony corpus, KK-RU LID/sentiment papers, Mendeley KZ news).

## Things to check before running on real data
1. Stage 1 prints each dataset's columns and warns if labels could not be mapped (KazSAnDRA label encoding must be confirmed on its card).
2. English sources are binary; Kazakh/Turkish have a neutral class. Use `stage5_split.py --binary` or add an English neutral source.
3. Zero-shot and joint are both evaluated on `target_test`, so target data used in joint training never leaks into evaluation.
