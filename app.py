import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="Starbucks Review Analyzer", page_icon="☕", layout="centered")

@st.cache_resource(show_spinner="Loading models... This may take a minute on first run.")
def load_models():
    sentiment_analyzer = pipeline(
        "text-classification",
        model="Evelyn1489/starbucks-sentiment-distilbert",
        device=-1
    )
    gen_model_name = "MBZUAI/LaMini-Flan-T5-248M"
    gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name)
    return sentiment_analyzer, gen_tokenizer, gen_model

sentiment_analyzer, gen_tokenizer, gen_model = load_models()

def analyze_sentiment(review):
    result = sentiment_analyzer(review, truncation=True, max_length=256)[0]
    return result["label"], result["score"]

def generate_summary(review):
    prompt = f"Summarize the following customer review in 1-2 sentences:\n\n{review}"
    inputs = gen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    output_ids = gen_model.generate(**inputs, max_length=80, num_beams=4, early_stopping=True)
    return gen_tokenizer.decode(output_ids[0], skip_special_tokens=True)

def generate_reply(review, sentiment, summary):
    if sentiment == "Negative":
        prompt = (
            f"Write a reply from Starbucks directly to a customer. "
            f"The customer complained: \"{summary}\" "
            f"Start with \"Dear valued customer,\" then apologize for their specific issues, "
            f"explain how you will fix those problems, "
            f"and offer them a free drink on their next visit. "
            f"End with \"Sincerely, Starbucks Customer Care Team\""
        )
    else:
        prompt = (
            f"Write a reply from Starbucks directly to a customer. "
            f"The customer praised: \"{summary}\" "
            f"Start with \"Dear valued customer,\" then thank them for the specific things they mentioned, "
            f"say you will share their kind words with the store team, "
            f"and invite them to try a new seasonal drink next time. "
            f"End with \"Sincerely, Starbucks Customer Care Team\""
        )
    inputs = gen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    output_ids = gen_model.generate(**inputs, max_length=200, num_beams=5, no_repeat_ngram_size=3, early_stopping=True)
    return gen_tokenizer.decode(output_ids[0], skip_special_tokens=True)

# UI
st.title("☕ Starbucks Review Analyzer")
st.markdown(
    "Analyze customer reviews using **3 AI pipelines**: "
    "Sentiment Analysis, Summarization, and Auto Service Reply."
)

review = st.text_area(
    "Enter a customer review:",
    height=150,
    placeholder="e.g., I waited 30 minutes for my coffee and it was cold when I got it..."
)

if st.button("🔍 Analyze", type="primary"):
    if review.strip():
        with st.spinner("Analyzing sentiment..."):
            sentiment, confidence = analyze_sentiment(review)
        with st.spinner("Generating summary..."):
            summary = generate_summary(review)
        with st.spinner("Generating service reply..."):
            reply = generate_reply(review, sentiment, summary)

        st.divider()

        st.subheader("Sentiment")
        if sentiment == "Negative":
            st.error(f"😞 {sentiment} (Confidence: {confidence:.1%})")
        else:
            st.success(f"😊 {sentiment} (Confidence: {confidence:.1%})")

        if confidence < 0.75:
            st.warning("⚠️ Low confidence — this review may be ambiguous. Please review manually.")

        st.subheader("Summary")
        st.info(summary)

        st.subheader("Suggested Service Reply")
        st.write(reply)
    else:
        st.warning("Please enter a review to analyze.")

st.divider()
st.caption(
    "ISOM5240 Group Project | "
    "Sentiment model: Fine-tuned DistilBERT | "
    "Summary & Reply model: LaMini-Flan-T5-248M"
)
