import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="Starbucks Review Analyzer", page_icon="☕", layout="centered")

@st.cache_resource(show_spinner="Loading models...")
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
    result = sentiment_analyzer(review)[0]
    return result["label"], result["score"]

def generate_summary(review):
    prompt = f"Summarize the following customer review in 1-2 sentences:\n\n{review}"
    inputs = gen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        output_ids = gen_model.generate(**inputs, max_length=80, num_beams=4, early_stopping=True)
    return gen_tokenizer.decode(output_ids[0], skip_special_tokens=True)

def generate_reply(review, sentiment, summary):
    if sentiment == "Negative":
        prompt = (
            f"Write a reply from Starbucks directly to a customer. "
            f"The customer complained: \"{summary}\" "
            f"Start with \"Dear valued customer,\" then apologize for their specific issues, "
            f"explain how you will fix those problems, "
            f"and offer a free drink on their next visit. "
            f"End with \"Sincerely, Starbucks Customer Care Team\""
        )
    else:
        prompt = (
            f"Write a reply from Starbucks directly to a customer. "
            f"The customer praised: \"{summary}\" "
            f"Start with \"Dear valued customer,\" then thank them for the specific things they liked, "
            f"say you will share their kind words with the store team, "
            f"and invite them to visit again soon. "
            f"End with \"Sincerely, Starbucks Customer Care Team\""
        )
    inputs = gen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        output_ids = gen_model.generate(**inputs, max_length=200, num_beams=5, no_repeat_ngram_size=3, early_stopping=True)
    return gen_tokenizer.decode(output_ids[0], skip_special_tokens=True)

if "sentiment_result" not in st.session_state:
    st.session_state.sentiment_result = None
if "confidence_result" not in st.session_state:
    st.session_state.confidence_result = None
if "summary_result" not in st.session_state:
    st.session_state.summary_result = ""
if "reply_result" not in st.session_state:
    st.session_state.reply_result = ""

st.title("☕ Starbucks Review Analyzer")
st.markdown("Analyze customer reviews using 3 AI pipelines: Sentiment Analysis, Summarization, and Auto Service Reply.")

review = st.text_area("Enter a customer review:", height=150)

if st.button("🔍 Analyze", type="primary"):
    if review.strip():
        with st.spinner("Analyzing sentiment..."):
            sentiment, confidence = analyze_sentiment(review)
            st.session_state.sentiment_result = sentiment
            st.session_state.confidence_result = confidence
        with st.spinner("Generating summary..."):
            summary = generate_summary(review)
            st.session_state.summary_result = summary
        with st.spinner("Generating service reply..."):
            reply = generate_reply(review, sentiment, summary)
            st.session_state.reply_result = reply

if st.session_state.sentiment_result is not None:
    st.divider()

    st.subheader("Sentiment")
    if st.session_state.sentiment_result == "Negative":
        st.error(f"😞 {st.session_state.sentiment_result} (Confidence: {st.session_state.confidence_result:.1%})")
    else:
        st.success(f"😊 {st.session_state.sentiment_result} (Confidence: {st.session_state.confidence_result:.1%})")

    st.subheader("Summary")
    st.info(st.session_state.summary_result)

    st.subheader("Suggested Service Reply")
    edited = st.text_area("Edit your reply:", value=st.session_state.reply_result, height=200, key="final_reply")

    safe_text = edited.replace("'", "\\'").replace('"', '\\"').replace("\n", "\\n")
    copy_button = f"""
    <button onclick="navigator.clipboard.writeText(`{edited}`); alert('Copied successfully!');"
    style="padding:10px 24px; background:#0071e3; color:white; border:none; border-radius:8px; font-size:16px; cursor:pointer;">
    📋 ONE-CLICK COPY
    </button>
    """
    st.components.v1.html(copy_button, height=60)

st.divider()
st.caption("ISOM5240 Group Project")
