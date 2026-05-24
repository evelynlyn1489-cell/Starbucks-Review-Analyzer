import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(
    page_title="Starbucks Review Analyzer",
    page_icon="https://upload.wikimedia.org/wikipedia/en/thumb/d/d3/Starbucks_Corporation_Logo_2011.svg/1200px-Starbucks_Corporation_Logo_2011.svg.png",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─── Starbucks-inspired CSS (brand greens, Sodo Sans spirit, clean cards) ───
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,700;1,9..40,400&family=Playfair+Display:wght@600;700&display=swap');

:root {
    --g1: #1E3932;
    --g2: #00704A;
    --g3: #00A862;
    --g4: #D4E9E2;
    --g5: #F1F8F6;
    --warm: #F9F6F2;
    --ink: #1E3932;
    --ink2: #3C6255;
    --ink3: #6B8F71;
    --white: #FFFFFF;
    --red: #D14836;
    --red-bg: #FDF0EE;
    --gold: #CBA258;
    --gold-bg: #FBF7EF;
}

/* ── Reset ── */
#MainMenu, header, footer {visibility:hidden}
.stApp {background:var(--warm);font-family:'DM Sans',sans-serif}
h1,h2,h3,h4 {font-family:'Playfair Display',serif !important;color:var(--ink) !important}

/* ── Hero ── */
.hero {
    background: var(--g1);
    margin: -6rem -4rem 0 -4rem;
    padding: 3rem 2rem 2.5rem;
    text-align:center;
    position:relative;
    overflow:hidden;
}
.hero::before {
    content:'';position:absolute;inset:0;
    background:
        radial-gradient(circle at 20% 50%, rgba(0,168,98,.12) 0%, transparent 50%),
        radial-gradient(circle at 80% 30%, rgba(0,112,74,.15) 0%, transparent 50%);
}
.hero-logo {
    width:72px;height:72px;border-radius:50%;
    border:2px solid rgba(255,255,255,.2);
    margin:0 auto 1rem;display:block;position:relative;z-index:1;
}
.hero h1 {
    color:var(--white) !important;font-size:1.9rem;margin:0;
    position:relative;z-index:1;letter-spacing:.3px;
}
.hero p {
    color:var(--g4);font-size:.92rem;margin:.4rem 0 0;
    font-family:'DM Sans',sans-serif;position:relative;z-index:1;
}

/* ── Pipeline pills ── */
.pills {
    display:flex;justify-content:center;gap:.6rem;
    margin:1.6rem auto 1.2rem;flex-wrap:wrap;
}
.pill {
    background:var(--white);
    border:1.5px solid var(--g4);
    color:var(--g1);
    padding:.35rem .9rem;border-radius:999px;
    font-size:.72rem;font-weight:500;letter-spacing:.5px;
    text-transform:uppercase;
}
.pill-dot {color:var(--g3);margin:0 .15rem;font-size:.6rem}

/* ── Cards ── */
.card {
    background:var(--white);
    border-radius:14px;
    padding:1.4rem 1.5rem;
    margin-bottom:.9rem;
    border:1px solid rgba(30,57,50,.06);
    box-shadow:0 1px 4px rgba(30,57,50,.04);
}

/* ── Section eyebrow ── */
.eyebrow {
    font-family:'DM Sans',sans-serif;
    font-size:.65rem;font-weight:700;
    text-transform:uppercase;letter-spacing:2px;
    color:var(--g2);margin-bottom:.6rem;
}

/* ── Sentiment chips ── */
.chip-neg {
    display:inline-flex;align-items:center;gap:.5rem;
    background:var(--red-bg);
    border:1px solid #F5C6C0;
    color:var(--red);
    padding:.65rem 1.2rem;border-radius:10px;
    font-size:1.05rem;font-weight:600;
}
.chip-pos {
    display:inline-flex;align-items:center;gap:.5rem;
    background:#EBF7EE;
    border:1px solid #A8DAAF;
    color:#1B7A2B;
    padding:.65rem 1.2rem;border-radius:10px;
    font-size:1.05rem;font-weight:600;
}
.chip-warn {
    display:inline-block;margin-top:.5rem;
    background:var(--gold-bg);border:1px solid #E8D5A8;
    color:#8B6914;padding:.4rem .9rem;border-radius:8px;
    font-size:.8rem;font-weight:500;
}

/* ── Summary strip ── */
.summary-strip {
    background:var(--g5);
    border-left:3px solid var(--g2);
    padding:.9rem 1.1rem;
    border-radius:0 10px 10px 0;
    color:var(--ink);font-size:.92rem;line-height:1.65;
}

/* ── Reply block ── */
.reply-block {
    background:var(--warm);
    border:1px solid #E8E4DD;
    padding:1rem 1.2rem;border-radius:10px;
    color:var(--ink);font-size:.92rem;line-height:1.65;
    white-space:pre-wrap;
}

/* ── Button ── */
div.stButton > button[kind="primary"] {
    background:var(--g1) !important;
    color:var(--white) !important;
    border:none !important;border-radius:10px !important;
    padding:.65rem 1.6rem !important;
    font-family:'DM Sans',sans-serif !important;
    font-size:.92rem !important;font-weight:600 !important;
    letter-spacing:.3px;width:100%;
    transition:background .2s,transform .15s;
}
div.stButton > button[kind="primary"]:hover {
    background:var(--g2) !important;
    transform:translateY(-1px);
}

/* ── Textarea ── */
.stTextArea textarea {
    border-radius:10px !important;
    border:1.5px solid var(--g4) !important;
    font-size:.9rem !important;padding:.9rem !important;
    background:var(--white) !important;
    font-family:'DM Sans',sans-serif !important;
}
.stTextArea textarea:focus {
    border-color:var(--g2) !important;
    box-shadow:0 0 0 3px rgba(0,112,74,.1) !important;
}

/* ── Footer ── */
.foot {
    text-align:center;color:var(--ink3);
    font-size:.7rem;padding:1.8rem 0 .8rem;
    letter-spacing:.3px;
}
.foot a {color:var(--g2);text-decoration:none}
</style>
""", unsafe_allow_html=True)


# ─── Models ───
@st.cache_resource(show_spinner="☕ Brewing your AI models...")
def load_models():
    sentiment_analyzer = pipeline(
        "text-classification",
        model="Evelyn1489/starbucks-sentiment-distilbert",
        device=-1, torch_dtype=torch.float32
    )
    gen_model_name = "MBZUAI/LaMini-Flan-T5-248M"
    gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(
        gen_model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    return sentiment_analyzer, gen_tokenizer, gen_model

sentiment_analyzer, gen_tokenizer, gen_model = load_models()


# ─── Functions ───
def analyze_sentiment(review):
    result = sentiment_analyzer(review)[0]
    return result["label"], result["score"]

def generate_summary(review):
    prompt = f"Summarize the following customer review in 1-2 sentences:\n\n{review}"
    inputs = gen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        out = gen_model.generate(**inputs, max_length=80, num_beams=4, early_stopping=True)
    return gen_tokenizer.decode(out[0], skip_special_tokens=True)

def generate_reply(review, sentiment, summary):
    if sentiment == "Negative":
        prompt = (f"Write a short apology letter to a Starbucks customer. "
                  f"The customer said: {summary} "
                  f"Apologize for these exact problems and offer a free drink.")
    else:
        prompt = (f"Write a short thank you letter to a Starbucks customer. "
                  f"The customer said: {summary} "
                  f"Thank them for these exact compliments and invite them back.")
    inputs = gen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        out = gen_model.generate(
            **inputs, max_length=200, num_beams=5,
            no_repeat_ngram_size=3, temperature=0.7, early_stopping=True
        )
    return gen_tokenizer.decode(out[0], skip_special_tokens=True)


# ─── State ───
for k in ["sentiment_result","confidence_result","summary_result","reply_result","review_input"]:
    if k not in st.session_state:
        st.session_state[k] = None if k in ("sentiment_result","confidence_result") else ""


# ─── Hero header ───
st.markdown("""
<div class="hero">
    <img class="hero-logo" src="https://upload.wikimedia.org/wikipedia/en/thumb/d/d3/Starbucks_Corporation_Logo_2011.svg/1200px-Starbucks_Corporation_Logo_2011.svg.png" alt="Starbucks">
    <h1>Review Analyzer</h1>
    <p>AI-powered customer feedback analysis</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="pills">
    <span class="pill">📊 Sentiment</span>
    <span class="pill-dot">●</span>
    <span class="pill">📝 Summary</span>
    <span class="pill-dot">●</span>
    <span class="pill">💬 Reply</span>
</div>
""", unsafe_allow_html=True)


# ─── Input ───
st.markdown('<div class="card">', unsafe_allow_html=True)
review = st.text_area(
    "Paste a customer review",
    height=130,
    value=st.session_state.review_input or "",
    placeholder="I waited 30 minutes for my coffee and it was cold when I got it. The barista was rude and got my order wrong…"
)
st.session_state.review_input = review
clicked = st.button("☕  Analyze Review", type="primary")
st.markdown('</div>', unsafe_allow_html=True)

if clicked:
    if review.strip():
        with st.spinner("Reading sentiment..."):
            s, c = analyze_sentiment(review)
            st.session_state.sentiment_result = s
            st.session_state.confidence_result = c
        with st.spinner("Summarizing..."):
            st.session_state.summary_result = generate_summary(review)
        with st.spinner("Drafting reply..."):
            st.session_state.reply_result = generate_reply(review, s, st.session_state.summary_result)
    else:
        st.warning("Please paste a review first.")


# ─── Results ───
if st.session_state.sentiment_result is not None:
    sent = st.session_state.sentiment_result
    conf = st.session_state.confidence_result

    # Sentiment
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Pipeline 1 · Sentiment Analysis</div>', unsafe_allow_html=True)
    if sent == "Negative":
        st.markdown(f'<div class="chip-neg">😞 Negative &nbsp;·&nbsp; {conf:.1%} confidence</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chip-pos">😊 Positive &nbsp;·&nbsp; {conf:.1%} confidence</div>', unsafe_allow_html=True)
    if conf < 0.75:
        st.markdown('<div class="chip-warn">⚠ Low confidence — this review may be ambiguous. Manual review recommended.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Summary
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Pipeline 2 · Review Summary</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="summary-strip">{st.session_state.summary_result}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Reply
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Pipeline 3 · Suggested Service Reply</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="reply-block">{st.session_state.reply_result}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Edit
    with st.expander("✏️ Edit & copy the reply"):
        edited = st.text_area("", value=st.session_state.reply_result, height=160, key="editor", label_visibility="collapsed")
        st.session_state.reply_result = edited
        st.code(edited, language=None)
        st.caption("↑ Click the copy icon in the top-right corner to copy.")


# ─── Footer ───
st.markdown("""
<div class="foot">
    ISOM5240 Group Project &nbsp;·&nbsp;
    Sentiment: Fine-tuned DistilBERT &nbsp;·&nbsp;
    Summary & Reply: LaMini-Flan-T5-248M<br>
    <a href="https://www.starbucks.com" target="_blank">Starbucks Corporation</a>
</div>
""", unsafe_allow_html=True)
