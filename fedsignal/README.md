# FedSignal — Central Bank NLP for Bond Yield Prediction

**HooHacks 2026 · Team project · March 21–22, 2026**
**My role:** backend algorithm contributor (sentiment model and yield regression)

FedSignal reads monetary policy statements from four central banks: the US Federal Reserve, Bank of England, Reserve Bank of Australia and State Bank of Pakistan. It scores how *hawkish* or *dovish* each statement is, then uses that score to estimate the direction of short-term government bond yields over the next 60 days.

---

## The problem

Bond markets move on how central banks *talk*, not just on what they do with rates. The same rate decision can push yields in different directions depending on whether the statement signals more tightening or an end to it. Human analysts read these statements by hand. We wanted to see whether a fine-tuned language model could turn that reading into a number, and whether the number carries any signal about where yields go next.

## How it works

```
Central bank websites ──► statements_pull.py ──► data/*.json
                                                     │
Hand-labeled sentences ──► fine-tuned DistilBERT ◄───┘
   (0 = dovish, 1 = hawkish)        │
                                    ▼
                      per-sentence hawkishness scores
                                    │
FRED macro + yield data ──► regression on sentiment + rate spreads
                                    │
                                    ▼
              predicted 60-day change in the 1-year yield
              (with an interval and an up/down direction)
```

1. **Data collection:** `src/statements_pull.py` scrapes policy statements from each central bank's website (HTML and PDF) and saves them under `data/`. `src/data_collector.py` pulls macroeconomic and yield-curve series from the FRED API.
2. **Sentiment model:** a `distilbert-base-uncased` model, fine-tuned in PyTorch with Hugging Face `Trainer` as a regression model. It learns from hand-labeled policy sentences scored from 0 (very dovish) to 1 (very hawkish). Each country gets its own model so it can learn that central bank's style of language.
3. **Statement scoring:** each statement is split into sentences (or 3-sentence chunks in the later version). Boilerplate such as voting lists and press contacts is filtered out, then each piece is scored and the scores are combined.
4. **Yield regression:** a linear regression relates the sentiment score and rate spreads to the 60-day change in the 1-year yield. It's evaluated with leave-one-out out-of-sample R² and directional accuracy.
5. **Serving:** `src/api.py` exposes the model as a FastAPI endpoint (`POST /analyze`). The team's Streamlit frontend calls this endpoint and isn't included here.

## How the model evolved

The notebooks show the model improving over three versions:

| Notebook | What changed |
|---|---|
| `01_us_fed_nlp_regression.ipynb` | US only. FRED data pipeline, FOMC statement scraping, a simple macro regime classifier, the first DistilBERT sentiment model and the regression |
| `02_multi_country_model.ipynb` | Extended to the UK, Australia and Pakistan, with one fine-tuned model per country and per-country training sentences written in each bank's style |
| `03_improved_multi_country_model.ipynb` | About 200 labeled sentences per country (up from about 70), change-in-sentiment and meeting-minutes features, paragraph-level scoring, an ensemble with a sentence-transformer similarity score, and a side-by-side comparison of feature sets |

## Tech stack

Python · PyTorch · Hugging Face Transformers (DistilBERT) · sentence-transformers · scikit-learn · pandas · FRED API (`fredapi`) · BeautifulSoup / pdfplumber · FastAPI

## Repository layout

```
fedsignal/
├── notebooks/        # model development, in order (01 → 03)
├── src/
│   ├── statements_pull.py   # central bank statement scraper and loader
│   ├── data_collector.py    # FRED macro / yield data pull
│   ├── analyze.py           # trains the sentiment models and scores a statement
│   └── api.py               # FastAPI endpoint
├── data/             # scraped statements (public central bank releases)
└── requirements.txt
```

## Running it

```bash
pip install -r requirements.txt
export FRED_API_KEY=your_key_here     # free key from fred.stlouisfed.org

# API
cd src && uvicorn api:app --port 8000
curl -X POST localhost:8000/analyze -H "Content-Type: application/json" \
     -d '{"country": "US", "statement": "The Committee decided to raise the target range..."}'

# Notebooks: run from the notebooks/ folder
jupyter notebook notebooks/
```

`analyze.py` fine-tunes its models when it first loads, which takes a few minutes on CPU.

## Limitations and what I'd improve

- **Small sample.** There are only a few dozen policy meetings per country, so the regression results are noisy and shouldn't be over-read.
- **Validation for time series.** Leave-one-out validation lets the model train on meetings that happened *after* the one it predicts. Walk-forward validation, which trains only on the past, would be a more honest test, and it's the first thing I'd change.
- **Label quality.** The training sentences were labeled by our team during the hackathon. Labels from several annotators, with a check on how often they agree, would make the sentiment scores more trustworthy.
- **Fixed coefficients in the API.** `analyze.py` uses regression coefficients copied from the notebook. A production version would save and load the fitted model instead.

---

*Built with my teammates at HooHacks 2026. The original team repository is private. This folder contains the backend and modeling code, with API keys removed.*
