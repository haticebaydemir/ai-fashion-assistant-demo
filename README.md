# 🤖 AI Fashion Assistant

> **Conversational Fashion Search System powered by AI**

⚠️ **Status:** Migrated to production microservice architecture

---

## 📊 Project Overview

Advanced fashion search and recommendation system using:
- **Semantic Search:** mpnet + CLIP embeddings
- **Vector Database:** FAISS (44K products)
- **LLM Integration:** Conversational interface
- **Ranking:** LightGBM learning-to-rank
- **Personalization:** Collaborative filtering (ALS)

---

## 🏗️ New Architecture (v2.0)

### **Microservice Design:**

```
┌─────────────────────────────────────┐
│   Streamlit Chat UI (Frontend)     │
│   - Conversational interface        │
│   - Product visualization           │
│   - Multi-turn dialogue             │
└─────────────────────────────────────┘
              ↓ HTTP REST
┌─────────────────────────────────────┐
│   FastAPI Backend (Microservice)    │
│   - FAISS vector search             │
│   - LLM chat integration            │
│   - Ranking & personalization       │
│   - Production-ready APIs           │
└─────────────────────────────────────┘
```

---

## ✨ Features

### **Search Capabilities:**
- ✅ Text search (semantic understanding)
- ✅ Image search (visual similarity)
- ✅ Hybrid search (text + image)
- ✅ Conversational search (LLM-powered)

### **Intelligence:**
- ✅ Query rewriting (+12% recall)
- ✅ Intent detection & slot extraction
- ✅ Multi-turn conversation
- ✅ Result explainability
- ✅ Personalized recommendations

### **Production:**
- ✅ FastAPI REST APIs
- ✅ Docker containerization
- ✅ Monitoring (Prometheus + Grafana)
- ✅ Comprehensive testing
- ✅ API documentation (OpenAPI)

---

## 📈 Performance

| Metric | Value | Baseline | Improvement |
|--------|-------|----------|-------------|
| **NDCG@10** | 86.6% | 71.2% | +15.4pp |
| **Recall@10** | 48.0% | 35.0% | +37.1% |
| **MRR** | 89.7% | 76.3% | +13.4pp |
| **Latency** | 87ms | 45ms | 1.9x |

---

## 🔧 Technology Stack

**Backend:**
- FastAPI (REST API)
- FAISS (vector search)
- PyTorch + Transformers
- LightGBM (ranking)
- Scikit-learn (personalization)

**Models:**
- sentence-transformers/paraphrase-multilingual-mpnet-base-v2
- openai/clip-vit-base-patch32
- Gemini 1.5 Flash (LLM)

**Frontend:**
- Streamlit (Chat UI)
- Gradio (Alternative)

**Infrastructure:**
- Docker + docker-compose
- ngrok (development tunneling)
- Prometheus + Grafana (monitoring)

---

## 🗂️ Dataset

**Fashion Product Images Dataset** (Kaggle)
- 44,446 products
- 7 master categories
- 3 genders
- 46 colors
- 4 seasons

---

## 🎓 Academic Context

**TÜBİTAK 2209-A Research Project**

**Research Areas:**
- Multimodal information retrieval
- Conversational AI
- Learning-to-rank
- Collaborative filtering
- Semantic search

---

## 🚧 Development Status

**Phase 1-2:** Data preparation + Embeddings ✅  
**Phase 3-4:** FAISS search + Evaluation ✅  
**Phase 5:** LightGBM ranking ✅  
**Phase 6:** Personalization (ALS) ✅  
**Phase 7:** Production APIs ✅  
**Phase 8:** LLM integration ✅  
**Phase 9-10:** Evaluation + Reproducibility ✅  

**Current:** Full-stack integration (FastAPI + Streamlit) 🔨

---

## 📝 Migration Notes

**Previous Version (v1.0):**
- Streamlit Cloud demo
- Monolithic architecture
- Limited scalability

**Current Version (v2.0):**
- Microservice architecture
- Production-ready
- Scalable & maintainable
- API-first design

---

## 🤝 Contributing

This is an active research project. For questions or collaboration opportunities, please open an issue.

---

## 📄 License

Academic research project - TÜBİTAK 2209-A

---

## 📧 Contact

For more information about this project, please reach out through GitHub issues.

---

**Built with** ❤️ **for the future of fashion e-commerce**

⭐ Star this repo if you find it interesting!

---

*Last updated: December 2024*
