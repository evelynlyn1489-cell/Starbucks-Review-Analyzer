import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# ─── Page config ───
st.set_page_config(
    page_title="Starbucks Review Analyzer",
    page_icon="☕",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─── Starbucks brand CSS ───
st.markdown("""
<style>
    /* === Starbucks brand colors === */
    :root {
        --sbx-green: #00704A;
        --sbx-green-dark: #006241;
        --sbx-green-light: #D4E9E2;
        --sbx-black: #27251F;
        --sbx-warm: #F2F0EB;
    }

    /* Hide default Streamlit header & footer */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Page background */
    .stApp {
        background-color: var(--sbx-warm);
    }

    /* Custom header bar */
    .sbx-header {
        background: linear-gradient(135deg, #006241 0%, #00704A 100%);
        padding: 1.8rem 2rem;
        border-radius: 0 0 20px 20px;
        margin: -1rem -1rem 2rem -1rem;
        text-align: center;
    }
    .sbx-header img {
        width: 60px;
        margin-bottom: 0.5rem;
    }
    .sbx-header h1 {
        color: white !important;
        font-size: 2rem;
        margin: 0;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .sbx-header p {
        color: #D4E9E2;
        font-size: 0.95rem;
        margin: 0.4rem 0 0 0;
    }

    /* Card container */
    .sbx-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    }

    /* Section labels */
    .sbx-label {
        color: var(--sbx-green-dark);
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.5rem;
    }

    /* Sentiment badges */
    .badge-neg {
        background: #FFF0F0;
        border: 1px solid #FFCDD2;
        color: #C62828;
        padding: 0.7rem 1.2rem;
        border-radius: 12px;
        font-size: 1.1rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-pos {
        background: #E8F5E9;
        border: 1px solid #A5D6A7;
        color: #2E7D32;
        padding: 0.7rem 1.2rem;
        border-radius: 12px;
        font-size: 1.1rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-warn {
        background: #FFF8E1;
        border: 1px solid #FFE082;
        color: #F57F17;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        font-size: 0.85rem;
        display: inline-block;
        margin-top: 0.5rem;
    }

    /* Summary box */
    .sbx-summary {
        background: var(--sbx-green-light);
        border-left: 4px solid var(--sbx-green);
        padding: 1rem 1.2rem;
        border-radius: 0 12px 12px 0;
        font-size: 0.95rem;
        color: var(--sbx-black);
        line-height: 1.6;
    }

    /* Reply box */
    .sbx-reply {
        background: #FAFAFA;
        border: 1px solid #E0E0E0;
        padding: 1rem 1.2rem;
        border-radius: 12px;
        font-size: 0.95rem;
        color: var(--sbx-black);
        line-height: 1.6;
        white-space: pre-wrap;
    }

    /* Analyze button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #006241 0%, #00704A 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.7rem 2rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(0,112,74,0.3) !important;
    }

    /* Text area */
    .stTextArea textarea {
        border-radius: 12px !important;
        border: 1.5px solid #D4E9E2 !important;
        font-size: 0.95rem !important;
        padding: 1rem !important;
    }
    .stTextArea textarea:focus {
        border-color: var(--sbx-green) !important;
        box-shadow: 0 0 0 2px rgba(0,112,74,0.15) !important;
    }

    /* Pipeline indicator */
    .pipeline-flow {
        display: flex;
        justify-content: center;
        gap: 0.5rem;
        align-items: center;
        margin: 1rem 0;
        flex-wrap: wrap;
    }
    .pipeline-step {
        background: white;
        border: 1.5px solid var(--sbx-green);
        color: var(--sbx-green-dark);
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .pipeline-arrow {
        color: var(--sbx-green);
        font-size: 0.9rem;
    }

    /* Footer */
    .sbx-footer {
        text-align: center;
        color: #9E9E9E;
        font-size: 0.75rem;
        padding: 1.5rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ─── Load models ───
@st.cache_resource(show_spinner="☕ Brewing your AI models... This may take a minute on first run.")
def load_models():
    sentiment_analyzer = pipeline(
        "text-classification",
        model="Evelyn1489/starbucks-sentiment-distilbert",
        device=-1,
        torch_dtype=torch.float32
    )
    gen_model_name = "MBZUAI/LaMini-Flan-T5-248M"
    gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(
        gen_model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    return sentiment_analyzer, gen_tokenizer, gen_model

sentiment_analyzer, gen_tokenizer, gen_model = load_models()


# ─── Analysis functions ───
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
            f"Write a short apology letter to a Starbucks customer. "
            f"The customer said: {summary} "
            f"Apologize for these exact problems and offer a free drink."
        )
    else:
        prompt = (
            f"Write a short thank you letter to a Starbucks customer. "
            f"The customer said: {summary} "
            f"Thank them for these exact compliments and invite them back."
        )
    inputs = gen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        output_ids = gen_model.generate(
            **inputs, max_length=200, num_beams=5,
            no_repeat_ngram_size=3, temperature=0.7, early_stopping=True
        )
    return gen_tokenizer.decode(output_ids[0], skip_special_tokens=True)


# ─── Session state ───
for key in ["sentiment_result", "confidence_result", "summary_result", "reply_result", "review_input"]:
    if key not in st.session_state:
        st.session_state[key] = None if "result" in key and "summary" not in key and "reply" not in key else ""


# ─── Header ───
st.markdown("""
<div class="sbx-header">
    <img src="https://upload.wikimedia.org/wikipedia/en/thumb/d/d3/Starbucks_Corporation_Logo_2011.svg/1200px-Starbucks_Corporation_Logo_2011.svg.png" alt="Starbucks Logo">
    <h1>Review Analyzer</h1>
    <p>AI-powered customer feedback analysis</p>
</div>
""", unsafe_allow_html=True)

# Pipeline flow indicator
st.markdown("""
<div class="pipeline-flow">
    <span class="pipeline-step">📊 Sentiment</span>
    <span class="pipeline-arrow">→</span>
    <span class="pipeline-step">📝 Summary</span>
    <span class="pipeline-arrow">→</span>
    <span class="pipeline-step">💬 Reply</span>
</div>
""", unsafe_allow_html=True)


# ─── Input section ───
st.markdown('<div class="sbx-card">', unsafe_allow_html=True)
review = st.text_area(
    "Paste a customer review below:",
    height=140,
    value=st.session_state.review_input or "",
    placeholder="e.g., I waited 30 minutes for my coffee and it was cold when I got it. The barista was rude and got my order wrong..."
)
st.session_state.review_input = review

analyze_clicked = st.button("☕ Analyze Review", type="primary")
st.markdown('</div>', unsafe_allow_html=True)

if analyze_clicked:
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
    else:
        st.warning("Please enter a review to analyze.")


# ─── Results ───
if st.session_state.sentiment_result is not None:

    # Sentiment
    st.markdown('<div class="sbx-card">', unsafe_allow_html=True)
    st.markdown('<div class="sbx-label">Pipeline 1 — Sentiment Analysis</div>', unsafe_allow_html=True)
    sentiment = st.session_state.sentiment_result
    confidence = st.session_state.confidence_result
    if sentiment == "Negative":
        st.markdown(f'<div class="badge-neg">😞 Negative — {confidence:.1%} confidence</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="badge-pos">😊 Positive — {confidence:.1%} confidence</div>', unsafe_allow_html=True)
    if confidence < 0.75:
        st.markdown('<div class="badge-warn">⚠️ Low confidence — this review may be ambiguous. Please review manually.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Summary
    st.markdown('<div class="sbx-card">', unsafe_allow_html=True)
    st.markdown('<div class="sbx-label">Pipeline 2 — Review Summary</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sbx-summary">{st.session_state.summary_result}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Reply
    st.markdown('<div class="sbx-card">', unsafe_allow_html=True)
    st.markdown('<div class="sbx-label">Pipeline 3 — Suggested Service Reply</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sbx-reply">{st.session_state.reply_result}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Editable reply
    with st.expander("✏️ Edit and customize the reply"):
        edited = st.text_area("Modify the reply below:", value=st.session_state.reply_result, height=180, key="reply_editor")
        st.session_state.reply_result = edited
        st.code(edited, language=None)
        st.caption("Click the copy icon (top-right of the box above) to copy.")


# ─── Footer ───
st.markdown("""
<div class="sbx-footer">
    ISOM5240 Group Project &nbsp;·&nbsp;
    Sentiment: Fine-tuned DistilBERT &nbsp;·&nbsp;
    Summary & Reply: LaMini-Flan-T5-248M<br>
    Starbucks Corporation &nbsp;·&nbsp; starbucks.com
</div>
""", unsafe_allow_html=True)
