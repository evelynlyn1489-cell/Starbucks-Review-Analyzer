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

# ─── Custom CSS Injection ───
st.markdown("""
<style>
    /* 全局背景色 (米色) */
    .stApp {
        background-color: #F3F0E6;
    }
    
    /* 全局文本颜色 */
    .stApp, .stApp p, .stApp span, .stApp label, .stApp div {
        color: #1E3932 !important;
    }

    /* 标题颜色 (深绿色) */
    h1, h2, h3, h4, h5, h6 {
        color: #1E3932 !important;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 800 !important;
    }

    /* 主按钮样式 (经典星巴克绿，圆角设计) */
    div.stButton > button:first-child {
        background-color: #006241 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 50px !important; /* 胶囊形状 */
        padding: 10px 24px !important;
        font-weight: bold !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    /* 按钮悬停效果 */
    div.stButton > button:first-child:hover {
        background-color: #1E3932 !important;
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
        transform: translateY(-1px);
    }

    /* 文本输入框样式 */
    .stTextArea textarea {
        background-color: #FFFFFF !important;
        border: 2px solid #D4D0C5 !important;
        border-radius: 8px !important;
        color: #1E3932 !important;
    }
    
    /* 文本输入框聚焦状态 */
    .stTextArea textarea:focus {
        border-color: #006241 !important;
        box-shadow: 0 0 0 1px #006241 !important;
    }

    /* 提示框/警告框样式调整 (使其更融于背景) */
    div[data-testid="stAlert"] {
        background-color: rgba(255, 255, 255, 0.7) !important;
        border-left: 5px solid #006241 !important;
        border-radius: 4px !important;
        color: #1E3932 !important;
    }

    /* 分割线颜色 */
    hr {
        border-color: #D4D0C5 !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── Load models (cached) ───
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    # Pipeline 1: Sentiment Analysis (fine-tuned DistilBERT)
    sentiment_analyzer = pipeline(
        "text-classification",
        model="Evelyn1489/starbucks-sentiment-distilbert",
        device=-1
    )

    # Pipeline 2 & 3: Summarization + Reply (LaMini-Flan-T5)
    gen_model_name = "MBZUAI/LaMini-Flan-T5-248M"
    gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name)

    return sentiment_analyzer, gen_tokenizer, gen_model

sentiment_analyzer, gen_tokenizer, gen_model = load_models()


# ─── Pipeline functions ───
def analyze_sentiment(review):
    """Pipeline 1: Classify review as Positive or Negative."""
    result = sentiment_analyzer(review)[0]
    return result["label"], result["score"]


def generate_summary(review):
    """Pipeline 2: Generate a 1-2 sentence summary of the review."""
    prompt = f"Summarize the following customer review in 1-2 sentences:\n\n{review}"
    inputs = gen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        output_ids = gen_model.generate(
            **inputs, max_length=80, num_beams=4, early_stopping=True
        )
    return gen_tokenizer.decode(output_ids[0], skip_special_tokens=True)


def generate_reply(review, sentiment, summary):
    """Pipeline 3: Generate a customer service reply based on sentiment."""
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


# ─── UI: Header ───
st.title("OUR IMPACT: Review Analyzer")
st.markdown(
    "Analyze customer reviews using 3 AI pipelines: "
    "**Sentiment Analysis, Summarization, and Auto Service Reply.**"
)


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
    st.divider()

    # Sentiment
    st.subheader("Sentiment")
    if st.session_state.sentiment_result == "Negative":
        st.error(f"😞 {st.session_state.sentiment_result} "
                 f"(Confidence: {st.session_state.confidence_result:.1%})")
    else:
        st.success(f"😊 {st.session_state.sentiment_result} "
                   f"(Confidence: {st.session_state.confidence_result:.1%})")

    # Summary
    st.subheader("Summary")
    st.info(st.session_state.summary_result)

    # Reply (editable)
    st.subheader("Suggested Service Reply")
    edited = st.text_area(
        "Edit your reply:",
        value=st.session_state.reply_result,
        height=200,
        key="final_reply"
    )

    # Copy button (browser clipboard API, no external libraries)
    safe_text = edited.replace("`", "\\`").replace("$", "\\$")
    copy_html = f"""
    <button onclick="navigator.clipboard.writeText(`{safe_text}`);
        this.innerText='✅ Copied!';
        setTimeout(()=>this.innerText='📋 Copy to Clipboard',2000);"
        style="padding:10px 24px; background:#006241; color:white;
        border:none; border-radius:50px; font-size:14px; font-weight:bold; 
        cursor:pointer; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        📋 Copy to Clipboard
    </button>
    """
    st.components.v1.html(copy_html, height=60)


# ─── UI: Footer ───
st.divider()
st.caption("ISOM5240 Group Project | Crafted with ☕")
