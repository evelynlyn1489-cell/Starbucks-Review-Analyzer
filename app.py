import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# ─── Page config ───
st.set_page_config(
    page_title="Starbucks Review Analyzer",
    page_icon="☕",
    layout="centered"
)

# ─── Custom CSS: Starbucks Brand Theme ───
st.markdown("""
<style>
    /* ── Import Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Lora:wght@400;500;600;700&family=Source+Sans+3:wght@300;400;500;600;700&display=swap');

    /* ── Starbucks Brand Colors ── */
    :root {
        --sb-green: #1E3932;
        --sb-green-light: #00704A;
        --sb-green-accent: #008248;
        --sb-cream: #F2F0EB;
        --sb-cream-dark: #E8E5DF;
        --sb-gold: #CBA258;
        --sb-text-dark: #1E3932;
        --sb-text-body: #3C3C3C;
        --sb-white: #FFFFFF;
        --sb-red-light: #FDEAEA;
        --sb-red: #D62B1E;
        --sb-blue-light: #EBF4F0;
    }

    /* ── Global Background ── */
    .stApp, .main, [data-testid="stAppViewContainer"] {
        background-color: var(--sb-cream) !important;
    }
    [data-testid="stHeader"] {
        background-color: var(--sb-cream) !important;
    }
    section[data-testid="stSidebar"] {
        background-color: var(--sb-cream-dark) !important;
    }

    /* ── Typography ── */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Lora', Georgia, serif !important;
        color: var(--sb-green) !important;
    }
    p, li, span, div, label, .stMarkdown {
        font-family: 'Source Sans 3', 'Segoe UI', sans-serif !important;
        color: var(--sb-text-body) !important;
    }

    /* ── Hero Banner ── */
    .hero-banner {
        background: linear-gradient(135deg, var(--sb-green) 0%, #2D5A4E 100%);
        border-radius: 16px;
        padding: 40px 36px 32px;
        margin-bottom: 32px;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -40px;
        right: -40px;
        width: 200px;
        height: 200px;
        background: rgba(255,255,255,0.04);
        border-radius: 50%;
    }
    .hero-banner h1 {
        color: var(--sb-white) !important;
        font-size: 2.2rem !important;
        margin-bottom: 8px !important;
        letter-spacing: -0.5px;
    }
    .hero-banner p {
        color: rgba(255,255,255,0.85) !important;
        font-size: 1.05rem;
        margin: 0;
        line-height: 1.6;
    }
    .hero-banner .accent-line {
        width: 48px;
        height: 3px;
        background: var(--sb-gold);
        border-radius: 2px;
        margin: 16px 0;
    }

    /* ── Card Containers ── */
    .card {
        background: var(--sb-white);
        border-radius: 12px;
        padding: 28px;
        margin-bottom: 20px;
        box-shadow: 0 1px 4px rgba(30,57,50,0.06);
        border: 1px solid rgba(30,57,50,0.06);
    }
    .card-header {
        font-family: 'Lora', Georgia, serif;
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--sb-green);
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .card-header .icon {
        font-size: 1.2rem;
    }

    /* ── Sentiment Badge ── */
    .sentiment-badge {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        padding: 12px 20px;
        border-radius: 10px;
        font-family: 'Source Sans 3', sans-serif;
        font-weight: 600;
        font-size: 1rem;
    }
    .sentiment-positive {
        background: var(--sb-blue-light);
        color: var(--sb-green);
        border: 1px solid rgba(0,112,74,0.15);
    }
    .sentiment-negative {
        background: var(--sb-red-light);
        color: var(--sb-red);
        border: 1px solid rgba(214,43,30,0.12);
    }

    /* ── Summary Box ── */
    .summary-box {
        background: var(--sb-cream);
        border-left: 3px solid var(--sb-gold);
        border-radius: 0 10px 10px 0;
        padding: 18px 22px;
        font-size: 0.98rem;
        line-height: 1.7;
        color: var(--sb-text-body);
    }

    /* ── Text Area Styling ── */
    .stTextArea textarea {
        background: var(--sb-white) !important;
        border: 1.5px solid var(--sb-cream-dark) !important;
        border-radius: 10px !important;
        font-family: 'Source Sans 3', sans-serif !important;
        font-size: 0.95rem !important;
        color: var(--sb-text-body) !important;
        padding: 16px !important;
        transition: border-color 0.2s ease;
    }
    .stTextArea textarea:focus {
        border-color: var(--sb-green-accent) !important;
        box-shadow: 0 0 0 2px rgba(0,130,72,0.1) !important;
    }

    /* ── Primary Button ── */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {
        background: var(--sb-green) !important;
        color: var(--sb-white) !important;
        border: none !important;
        border-radius: 24px !important;
        padding: 10px 32px !important;
        font-family: 'Source Sans 3', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        letter-spacing: 0.3px;
        transition: all 0.2s ease !important;
    }
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {
        background: var(--sb-green-light) !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(30,57,50,0.2) !important;
    }

    /* ── Spinner ── */
    .stSpinner > div {
        border-top-color: var(--sb-green-accent) !important;
    }

    /* ── Divider ── */
    hr {
        border-color: var(--sb-cream-dark) !important;
        opacity: 0.6;
    }

    /* ── Footer ── */
    .footer-text {
        text-align: center;
        font-family: 'Source Sans 3', sans-serif;
        font-size: 0.82rem;
        color: #9B9B9B;
        padding-top: 8px;
    }

    /* ── Hide default Streamlit elements ── */
    [data-testid="stInfo"], [data-testid="stError"], [data-testid="stSuccess"] {
        display: none;
    }

    /* ── Pipeline Step Indicators ── */
    .pipeline-steps {
        display: flex;
        gap: 8px;
        margin-bottom: 24px;
    }
    .pipeline-step {
        flex: 1;
        background: var(--sb-white);
        border: 1px solid var(--sb-cream-dark);
        border-radius: 8px;
        padding: 10px 14px;
        text-align: center;
        font-family: 'Source Sans 3', sans-serif;
        font-size: 0.82rem;
        color: var(--sb-text-body);
        font-weight: 500;
    }
    .pipeline-step .step-num {
        display: block;
        font-size: 0.7rem;
        color: var(--sb-green-accent);
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 2px;
    }
</style>
""", unsafe_allow_html=True)


# ─── Load models (cached) ───
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


# ─── Pipeline functions ───
def analyze_sentiment(review):
    result = sentiment_analyzer(review)[0]
    return result["label"], result["score"]


def generate_summary(review):
    prompt = f"Summarize the following customer review in 1-2 sentences:\n\n{review}"
    inputs = gen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        output_ids = gen_model.generate(
            **inputs, max_length=80, num_beams=4, early_stopping=True
        )
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
        output_ids = gen_model.generate(
            **inputs, max_length=200, num_beams=5,
            no_repeat_ngram_size=3, early_stopping=True
        )
    return gen_tokenizer.decode(output_ids[0], skip_special_tokens=True)


# ─── Session state ───
if "sentiment_result" not in st.session_state:
    st.session_state.sentiment_result = None
if "confidence_result" not in st.session_state:
    st.session_state.confidence_result = None
if "summary_result" not in st.session_state:
    st.session_state.summary_result = ""
if "reply_result" not in st.session_state:
    st.session_state.reply_result = ""


# ─── UI: Hero Banner ───
st.markdown("""
<div class="hero-banner">
    <h1>☕ Starbucks Review Analyzer</h1>
    <div class="accent-line"></div>
    <p>Analyze customer reviews using 3 AI pipelines: Sentiment Analysis, Summarization, and Auto Service Reply.</p>
</div>
""", unsafe_allow_html=True)

# ─── UI: Pipeline Steps ───
st.markdown("""
<div class="pipeline-steps">
    <div class="pipeline-step">
        <span class="step-num">Pipeline 1</span>
        Sentiment Analysis
    </div>
    <div class="pipeline-step">
        <span class="step-num">Pipeline 2</span>
        Summarization
    </div>
    <div class="pipeline-step">
        <span class="step-num">Pipeline 3</span>
        Service Reply
    </div>
</div>
""", unsafe_allow_html=True)


# ─── UI: Input ───
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


# ─── UI: Results ───
if st.session_state.sentiment_result is not None:
    st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)

    # Sentiment Card
    if st.session_state.sentiment_result == "Negative":
        badge_class = "sentiment-negative"
        emoji = "😞"
    else:
        badge_class = "sentiment-positive"
        emoji = "😊"

    st.markdown(f"""
    <div class="card">
        <div class="card-header"><span class="icon">📊</span> Sentiment Analysis</div>
        <div class="sentiment-badge {badge_class}">
            {emoji} {st.session_state.sentiment_result} &nbsp;·&nbsp; Confidence: {st.session_state.confidence_result:.1%}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Summary Card
    st.markdown(f"""
    <div class="card">
        <div class="card-header"><span class="icon">📝</span> Review Summary</div>
        <div class="summary-box">{st.session_state.summary_result}</div>
    </div>
    """, unsafe_allow_html=True)

    # Reply Card
    st.markdown("""
    <div class="card" style="padding-bottom: 8px;">
        <div class="card-header"><span class="icon">💬</span> Suggested Service Reply</div>
    </div>
    """, unsafe_allow_html=True)

    edited = st.text_area(
        "Edit your reply before sending:",
        value=st.session_state.reply_result,
        height=200,
        key="final_reply"
    )

    # Copy button
    safe_text = edited.replace("`", "\\`").replace("$", "\\$")
    copy_html = f"""
    <button onclick="navigator.clipboard.writeText(`{safe_text}`);
        this.innerText='✅ Copied!';
        setTimeout(()=>this.innerText='📋 Copy to Clipboard',2000);"
        style="padding:10px 24px; background:#1E3932; color:white;
        border:none; border-radius:20px; font-size:14px; cursor:pointer;
        font-family:'Source Sans 3',sans-serif; font-weight:600;
        transition: background 0.2s;">
        📋 Copy to Clipboard
    </button>
    """
    st.components.v1.html(copy_html, height=55)


# ─── UI: Footer ───
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<p class="footer-text">ISOM5240 Group Project &nbsp;·&nbsp; Powered by Hugging Face Transformers</p>', unsafe_allow_html=True)
