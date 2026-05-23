import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)


# --------------------------
# 页面设置
# --------------------------
st.set_page_config(
    page_title="Starbucks Review Analyzer",
    page_icon="☕",
    layout="centered"
)


# --------------------------
# 加载模型（只加载一次，缓存起来）
# --------------------------
@st.cache_resource(show_spinner="Loading models... This may take a minute on first run.")
def load_models():
    # Pipeline 1: 情感分析（你微调的 DistilBERT 模型）
    sentiment_analyzer = pipeline(
        "text-classification",
        model="Evelyn1489/starbucks-sentiment-distilbert",
        device=-1
    )

    # Pipeline 2 & 3: 摘要 + 自动回复（LaMini-Flan-T5 模型）
    gen_model_name = "MBZUAI/LaMini-Flan-T5-248M"
    gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name)

    return sentiment_analyzer, gen_tokenizer, gen_model


sentiment_analyzer, gen_tokenizer, gen_model = load_models()


# --------------------------
# 分析函数
# --------------------------
def analyze_sentiment(review):
    """Pipeline 1: 用微调后的 DistilBERT 判断好评/差评"""
    result = sentiment_analyzer(review, truncation=True, max_length=256)[0]
    label = result["label"]
    score = result["score"]
    return label, score


def generate_summary(review):
    """Pipeline 2: 用 LaMini-Flan-T5 生成评论摘要"""
    prompt = f"Summarize the following customer review in 1-2 sentences:\n\n{review}"
    inputs = gen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    output_ids = gen_model.generate(
        **inputs,
        max_length=80,
        num_beams=4,
        early_stopping=True
    )
    summary = gen_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return summary


def generate_reply(review, sentiment, summary):
    """Pipeline 3: 用 LaMini-Flan-T5 生成针对性的客服回复"""
    if sentiment == "Negative":
        prompt = (
            f"You are a Starbucks customer service manager. "
            f"A customer left this negative review: \"{summary}\" "
            f"Write a specific, personalized reply that: "
            f"1) Apologizes for the specific issues they mentioned, "
            f"2) Explains what Starbucks will do to fix those specific problems, "
            f"3) Offers a concrete gesture like a complimentary drink. "
            f"Do not use generic phrases. Address their exact complaints."
        )
    else:
        prompt = (
            f"You are a Starbucks customer service manager. "
            f"A customer left this positive review: \"{summary}\" "
            f"Write a specific, personalized thank-you reply that: "
            f"1) Mentions the specific things they praised, "
            f"2) Shares their feedback with the team or barista they mentioned, "
            f"3) Invites them to try a new seasonal drink on their next visit. "
            f"Do not use generic phrases. Reference their exact compliments."
        )

    inputs = gen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    output_ids = gen_model.generate(
        **inputs,
        max_length=200,
        num_beams=5,
        no_repeat_ngram_size=3,
        early_stopping=True
    )
    reply = gen_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return reply


# --------------------------
# 界面
# --------------------------
st.title("☕ Starbucks Review Analyzer")
st.markdown(
    "Analyze customer reviews using **3 AI pipelines**: "
    "Sentiment Analysis, Summarization, and Auto Service Reply."
)

# 输入区域
review = st.text_area(
    "Enter a customer review:",
    height=150,
    placeholder="e.g., I waited 30 minutes for my coffee and it was cold when I got it..."
)

# 分析按钮
if st.button("🔍 Analyze", type="primary"):
    if review.strip():
        # Pipeline 1: 情感分析
        with st.spinner("Analyzing sentiment..."):
            sentiment, confidence = analyze_sentiment(review)

        # Pipeline 2: 摘要
        with st.spinner("Generating summary..."):
            summary = generate_summary(review)

        # Pipeline 3: 自动回复（把摘要传进去，让回复更针对性）
        with st.spinner("Generating service reply..."):
            reply = generate_reply(review, sentiment, summary)

        # 显示结果
        st.divider()

        # 情感结果
        st.subheader("Sentiment")
        if sentiment == "Negative":
            st.error(f"😞 {sentiment} (Confidence: {confidence:.1%})")
        else:
            st.success(f"😊 {sentiment} (Confidence: {confidence:.1%})")

        # 摘要
        st.subheader("Summary")
        st.info(summary)

        # 客服回复
        st.subheader("Suggested Service Reply")
        st.write(reply)

    else:
        st.warning("Please enter a review to analyze.")

# 页脚
st.divider()
st.caption(
    "ISOM5240 Group Project | "
    "Sentiment model: Fine-tuned DistilBERT | "
    "Summary & Reply model: LaMini-Flan-T5-248M"
)
