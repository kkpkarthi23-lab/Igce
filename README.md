# 🎓 IGCE College Assistant

A simple RAG chatbot for **Indra Ganesan College of Engineering (IGCE)**.

It answers questions about admissions, courses, fees, placements, contact details, and facilities using a knowledge base.

## Features
- Semantic search (works in English & Tamil)
- Answers only from college knowledge base
- No API key required
- Fast and lightweight

## How to run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud
1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Create a new app and select this repository
4. Main file: `app.py`
5. Deploy

## Files
- `app.py` – Main Streamlit application
- `knowledge_base.json` – College information
- `embeddings.npy` – Pre-computed embeddings (for faster startup)
- `requirements.txt` – Python dependencies
