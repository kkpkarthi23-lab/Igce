"""
IGCE College Chatbot (RAG) - lightweight version
Answers only from knowledge_base.json using semantic search.
No heavy generative model → starts fast and works reliably.
"""
import json
import os
import traceback

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

KB_FILE = "knowledge_base.json"
EMB_FILE = "embeddings.npy"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 3
MIN_SCORE = 0.28


@st.cache_resource
def load_resources():
    if not os.path.exists(KB_FILE):
        st.error(f"❌ `{KB_FILE}` not found in the same folder.")
        st.stop()

    with open(KB_FILE, encoding="utf-8") as f:
        kb = json.load(f)

    if not kb:
        st.error("❌ Knowledge base is empty.")
        st.stop()

    model = SentenceTransformer(EMBED_MODEL)

    if os.path.exists(EMB_FILE):
        emb = np.load(EMB_FILE)
        if emb.shape[0] != len(kb):
            emb = model.encode(
                [c["text"] for c in kb],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            np.save(EMB_FILE, emb)
    else:
        emb = model.encode(
            [c["text"] for c in kb],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        np.save(EMB_FILE, emb)

    return kb, model, emb


def retrieve(question, kb, model, emb):
    q = model.encode([question], normalize_embeddings=True)[0]
    scores = emb @ q
    top_idx = np.argsort(scores)[::-1][:TOP_K]
    return [(kb[i], float(scores[i])) for i in top_idx]


def make_answer(question, hits):
    if not hits or hits[0][1] < MIN_SCORE:
        return (
            "I don't have that information in my knowledge base.\n\n"
            "Please check the official website **[igceng.com](https://www.igceng.com/)** "
            "or contact the college office."
        )

    # Build a clear answer from the top chunks
    parts = []
    for h, score in hits:
        if score < MIN_SCORE:
            continue
        parts.append(f"**{h['title']}**\n\n{h['text']}")

    if not parts:
        return (
            "I don't have a clear answer for that.\n\n"
            "Please visit **[igceng.com](https://www.igceng.com/)** for the latest information."
        )

    answer = "\n\n---\n\n".join(parts)
    answer += "\n\n*Source: Official IGCE knowledge base / igceng.com*"
    return answer


# ---------- UI ----------
st.set_page_config(
    page_title="IGCE Assistant",
    page_icon="🎓",
    layout="centered",
)

st.title("🎓 IGCE College Assistant")
st.caption(
    "Ask about admissions, courses, fees, placements, contact & facilities. "
    "தமிழிலும் கேட்கலாம்."
)

try:
    kb, embed_model, embeddings = load_resources()
except Exception as e:
    st.error(f"Failed to load resources: {e}")
    st.code(traceback.format_exc())
    st.stop()

if "chat" not in st.session_state:
    st.session_state.chat = []

with st.sidebar:
    st.header("Info")
    st.write(f"Knowledge chunks: **{len(kb)}**")
    st.write(f"Model: `{EMBED_MODEL}`")
    if st.button("🗑️ Clear chat"):
        st.session_state.chat = []
        st.rerun()
    st.markdown("---")
    st.markdown("[Official website](https://www.igceng.com/)")

# Show history
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Input
if q := st.chat_input("Ask a question about the college..."):
    with st.chat_message("user"):
        st.markdown(q)
    st.session_state.chat.append({"role": "user", "content": q})

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            try:
                hits = retrieve(q, kb, embed_model, embeddings)
                reply = make_answer(q, hits)
            except Exception as e:
                reply = f"⚠️ Error: {type(e).__name__}: {e}"

        st.markdown(reply)

        if hits and hits[0][1] >= MIN_SCORE:
            with st.expander("Matching scores"):
                for h, score in hits:
                    st.write(f"- {h['title']}: **{score:.2f}**")

    st.session_state.chat.append({"role": "assistant", "content": reply})
