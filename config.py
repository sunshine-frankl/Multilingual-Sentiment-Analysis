"""Shared configuration for the Data Collection Process (DCP).

Every source is a plain dict. `enabled=False` sources are NOT downloaded by default
(license unclear, gated, or needs manual access). Verify licences before publishing.
Unified schema used by all stages:
    id, text, label (neg/neu/pos or empty), label_raw, lang, domain, source, tier, license
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
CLEAN = DATA / "clean"
SPLITS = DATA / "splits"
ANNOT = DATA / "annotation"
for _p in (RAW, CLEAN, SPLITS, ANNOT):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 42

# ---------------------------------------------------------------------------
# Source registry  (tier numbering follows Section 3 of the DCP paper)
# ---------------------------------------------------------------------------
SOURCES = [
    # ---- Tier 1: Global / English source domain -------------------------------------------
    dict(key="sst2", tier=1, lang="en", domain="movie", enabled=True,
         hf_id="stanfordnlp/sst2", splits=["train", "validation"], limit=None,
         license="CC BY-SA 4.0 (check card)", note="binary; test labels hidden so not used"),
    dict(key="yelp_polarity", tier=1, lang="en", domain="restaurant", enabled=True,
         hf_id="fancyzhx/yelp_polarity", splits=["train"], limit=100_000, streaming=True,
         license="Yelp dataset terms (research use)", note="binary; first N rows"),
    dict(key="amazon_polarity", tier=1, lang="en", domain="product", enabled=True,
         hf_id="fancyzhx/amazon_polarity", splits=["train"], limit=100_000, streaming=True,
         license="Apache-2.0 (card)", note="binary; title+content merged"),

    # ---- Tier 2: Regional target-language sentiment ----------------------------------------
    dict(key="kazsandra", tier=2, lang="kk", domain="reviews", enabled=True,
         hf_id="issai/kazsandra", splits=None, limit=None, license="CC BY-SA/BY 4.0 (card: cc-by-4.0)",
         note="GATED on HF: accept terms on the dataset page, then `hf auth login`. 180k Kazakh reviews"),
    dict(key="tr_winvoker", tier=2, lang="tr", domain="reviews+tweets", enabled=True,
         hf_id="winvoker/turkish-sentiment-analysis-dataset", splits=None, limit=None,
         license="see HF card", note="product/store reviews + tweets + (neutral) Wikipedia sentences, 3-class",
         exclude_domain_regex="wiki"),     # rows whose `dataset` column matches are dropped (Wikipedia is not sentiment text)
    dict(key="tr_trsav1", tier=2, lang="tr", domain="e-commerce", enabled=True,
         hf_id="maydogan/Turkish_SentimentAnalysis_TRSAv1", splits=None, limit=None,
         license="see HF card", note="150k balanced 3-class e-commerce reviews"),
    dict(key="tr_product_reviews", tier=2, lang="tr", domain="product", enabled=False,
         hf_id="turkish_product_reviews", splits=None, limit=None,
         license="NO KNOWN LICENSE", note="235k, binary, heavily imbalanced; disabled by default"),

    # ---- Tier 3: Code-switching / noise probes ---------------------------------------------
    dict(key="krcs", tier=3, lang="kk-ru", domain="social", enabled=True,
         hf_id="BorisovMaksim/kk_ru_csw", splits=None, limit=None, license="see HF card",
         text_col="Original",              # columns are Original / Kazakh / Russian
         note="618 code-switched KK-RU sentences, NO sentiment labels (probe set only)"),

    # ---- Tier 4: Unlabelled web / news corpora (domain adaptation) -------------------------
    dict(key="kk_multidomain_leipzig", tier=4, lang="kk", domain="web", enabled=True,
         hf_id="kz-transformers/multidomain-kazakh-dataset", splits=["train"], limit=200_000,
         streaming=True, data_files="leipzig.csv", license="see HF card",
         note="unlabelled; the repo also has cc100 / oscar / news / books CSVs (multi-GB)"),
]

# Sources that cannot be downloaded automatically (documented in README, used manually)
MANUAL_SOURCES = [
    ("SarcasTurk (Turkish sarcasm, 1,515 Eksi Sozluk entries)",
     "On request from authors (ACL Anthology 2026.sigturk-1.6)"),
    ("Multilingual irony corpus (AR/FR/EN)", "https://github.com/bilalghanem/multilingual_irony"),
    ("Turkish Satirical News dataset (Zaytung vs Anadolu Agency)", "arXiv:2412.09247 (see paper for release link)"),
    ("Kazakh-Russian code-switch LID gold set + mixed sentiment pool", "arXiv:2608.00581 (check paper for release)"),
    ("Kazakhstani movie reviews 100k (RU/KK/code-switched)", "arXiv:2605.08600 (check paper for release)"),
    ("Kazakhstani & Russian news corpus (6.2M docs)", "https://data.mendeley.com/datasets/2vz7vtbhn2"),
]

TEXT_CANDS = ["text", "sentence", "review", "content", "comment", "tweet", "kk", "source"]
LABEL_CANDS = ["label", "sentiment", "polarity", "score", "rating", "stars"]

_NEG = {"neg", "negative", "negatif", "olumsuz", "0_neg"}
_POS = {"pos", "positive", "pozitif", "olumlu", "1_pos"}
_NEU = {"neu", "neutral", "notr", "nötr", "tarafsız", "tarafsiz", "nötür"}


def norm_label(v):
    """Map a raw label to neg/neu/pos, or '' if it cannot be mapped with confidence."""
    s = str(v).strip().lower()
    if s in _NEG:
        return "neg"
    if s in _POS:
        return "pos"
    if s in _NEU:
        return "neu"
    return ""
