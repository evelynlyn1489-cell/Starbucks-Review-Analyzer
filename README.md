# Starbucks Review Analyzer

## General Information

- **Project Title:** Starbucks Customer Review Sentiment Analysis Application
- **Description:** An AI-powered web application that analyzes Starbucks customer reviews through three deep learning pipelines: **Sentiment Analysis**, **Text Summarization**, and **Automated Service Reply Generation**, enabling data-driven customer experience management.

Built as part of the ISOM5240 Group Project.

## Deployed Application

- **Live App:** https://starbucks-review-analyzer-hdjyt6xktezyjxlkxpayqw.streamlit.app/
- **Fine-tuned Model:** https://huggingface.co/Evelyn1489/starbucks-sentiment-distilbert
- **GitHub Repository:** https://github.com/evelynlyn1489-cell/Starbucks-Review-Analyzer

## What It Does

Paste any Starbucks customer review into the app, and it will:

1. **Classify the sentiment** as Positive or Negative with a confidence score
2. **Summarize the review** into 1-2 key sentences
3. **Generate a service reply** tailored to the specific feedback

> Reviews with low confidence scores (< 75%) are flagged as ambiguous, prompting manual review.

## Live Demo

🔗 **[Try the app on Streamlit Cloud](YOUR_STREAMLIT_URL_HERE)**

## How It Works

```
Customer Review
       │
       ▼
┌──────────────────────────────┐
│  Pipeline 1: Sentiment       │  Fine-tuned DistilBERT
│  → Positive / Negative       │  (Evelyn1489/starbucks-sentiment-distilbert)
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Pipeline 2: Summarization   │  LaMini-Flan-T5-248M
│  → 1-2 sentence summary      │  (MBZUAI/LaMini-Flan-T5-248M)
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Pipeline 3: Auto Reply      │  LaMini-Flan-T5-248M
│  → Personalized response     │  (MBZUAI/LaMini-Flan-T5-248M)
└──────────────────────────────┘
```

## Models Used

| Pipeline | Model | Source | Fine-tuned? |
|----------|-------|--------|-------------|
| Sentiment Analysis | DistilBERT | [Hugging Face](https://huggingface.co/Evelyn1489/starbucks-sentiment-distilbert) | Yes |
| Text Summarization | LaMini-Flan-T5-248M | [Hugging Face](https://huggingface.co/MBZUAI/LaMini-Flan-T5-248M) | No |
| Auto Service Reply | LaMini-Flan-T5-248M | [Hugging Face](https://huggingface.co/MBZUAI/LaMini-Flan-T5-248M) | No |

## Dataset

- **Source:** Kaggle — Starbucks customer reviews
- **Raw data:** 850 reviews with 1-5 star ratings
- **After cleaning:** 703 reviews (removed empty reviews, missing ratings, duplicates)
- **Labels:** Binary classification — Negative (1-3 stars) / Positive (4-5 stars)
- **Training set:** 768 samples (balanced via oversampling)
- **Test accuracy:** 94.06%

## Tech Stack

- **Python** — Core language
- **Hugging Face Transformers** — Pre-trained models and fine-tuning
- **Streamlit** — Web application framework
- **PyTorch** — Deep learning backend
- **scikit-learn** — Data splitting and evaluation metrics

## Project Structure

```
├── app.py                # Streamlit application
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Run Locally

```bash
# Clone the repository
git clone https://github.com/evelynlyn1489-cell/Starbucks-Review-Analyzer.git
cd Starbucks-Review-Analyzer

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for pre-trained models and hosting
- [Streamlit](https://streamlit.io/) for the web app framework
- ISOM5240 course at HKUST
