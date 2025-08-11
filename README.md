# AI Developer Test — Option A: Smart Product Search (NLP)

This is a small e-commerce product catalog demo with a **natural language search** feature.  
Type things like:

- `running shoes under $100 with good reviews`
- `waterproof boots rating >= 4`
- `between $20 and $50 socks`
- `electronics above $50`

The query is **parsed** (price/rating/category) and **scored semantically** using a lightweight TF‑IDF + cosine similarity model that runs entirely in the browser (no backend or API key required).

## Features
- Product catalog (20 sample items) with category/price/rating filters
- **NLP search** with:
  - rule-based parsing of *price range*, *minimum rating*, and *category* (with synonyms, e.g., `sneakers` → `Shoes`)
  - TF‑IDF semantic matching over product name/description/category
- Sort by relevance, price, or rating
- Minimal modern UI (dark, responsive)

---

## How to Run (Windows / VS Code)

1. Install Node.js 18+.
2. In VS Code Terminal:
   ```bash
   cd ecom-nlp-search
   npm install
   npm run dev
   ```
3. Open the local URL printed by Vite (usually `http://localhost:5173`).

> No API keys needed. Everything runs locally in the browser.

---

## AI Feature Choice
**Option A – Smart Product Search (NLP).**  
We combine:
- **Parsing** natural language queries to extract filters (price min/max, rating min, category via synonyms).
- **Semantic scoring** using TF‑IDF vectors and cosine similarity, computed on the fly for the small demo catalog.

This is intentionally simple, fast, and fully local to satisfy test constraints.

### Possible Extensions
- Swap TF‑IDF for **embeddings** (e.g., small local models or OpenAI embeddings) and store vectors per product.  
- Add **typo tolerance** (e.g., Levenshtein distance) and query expansion via synonyms.  
- Log anonymized queries to improve spelling dictionaries and synonyms over time.

---

## Tools / Libraries
- React 18 + Vite + TypeScript
- No CSS framework to keep the repo minimal (custom styles inline)
- No server required

---

## Notable Assumptions
- Small demo catalog (20 items) ships as JSON.
- The NLP is intentionally lightweight; for production we’d use quality embeddings and a vector index (FAISS/pgvector).

---
