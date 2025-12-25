# 🎨 AI Fashion Assistant - Production Demo

> **Multimodal fashion search powered by AI**  
> NDCG@10: 86.6% | Recall@10: 48% | Latency: <100ms

A production-grade fashion search system that combines semantic text understanding (mpnet), visual similarity (CLIP), and fast vector search (FAISS) to deliver relevant product recommendations.

## 🚀 Live Demo

**Try it now:** [AI Fashion Assistant](https://ai-fashion-assistant-demo.streamlit.app)

## ✨ Features

### 🔍 **Text Search**
- **Semantic understanding**: "casual summer dress" → understands intent
- **Multilingual support**: Works in English & Turkish
- **Smart filtering**: Category, color, gender filters
- **Explainable results**: See why each product matched

### 📸 **Visual Search**
- **Image-based search**: Upload a photo, find similar products
- **CLIP-powered**: State-of-art vision-language model
- **Visual similarity**: Matches style, color, and pattern

### 🎯 **Performance**
- **Fast**: <100ms search time
- **Accurate**: 86.6% NDCG@10, 48% Recall@10
- **Scalable**: Handles 44K+ products efficiently

### 🧠 **Technologies**
- `mpnet`: Multilingual semantic embeddings (768d)
- `CLIP`: Vision-language alignment (512d text + 768d image)
- `FAISS`: Fast similarity search
- `Streamlit`: Interactive web interface

## 📊 Performance Metrics

| Metric | Value | Baseline (BM25) | Improvement |
|--------|-------|-----------------|-------------|
| NDCG@10 | 86.6% | 71.2% | +15.4pp |
| Recall@10 | 48.0% | 35.0% | +37.1% |
| MRR | 89.7% | 76.3% | +13.4pp |
| Latency | 87ms | 45ms | 1.9x |

## 🗂️ Dataset

**Fashion Product Images Dataset** (Kaggle)
- 44,446 products
- 7 categories: Topwear, Bottomwear, Shoes, Watches, Bags, etc.
- 3 genders: Men, Women, Unisex
- 46 colors, 4 seasons, 9 usage types

## 🏗️ Architecture

```
User Query
    ↓
Query Understanding
    ↓
Embedding Generation (mpnet + CLIP)
    ↓
FAISS Vector Search
    ↓
Results Ranking
    ↓
Personalized Results
```

## 💻 Local Development

### Prerequisites
- Python 3.10+
- 8GB RAM minimum
- CUDA (optional, for GPU acceleration)

### Installation

```bash
# Clone repository
git clone https://github.com/haticebaydemir/ai-fashion-assistant-demo.git
cd ai-fashion-assistant-demo

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run streamlit_app.py
```

### File Structure

```
ai-fashion-assistant-demo/
├── streamlit_app.py           # Main application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── .streamlit/
    └── config.toml            # Streamlit configuration
```

## 🔧 Configuration

The app expects the following directory structure in Google Drive:

```
/content/drive/MyDrive/
├── ai_fashion_assistant_v2/
│   ├── data/processed/
│   │   └── meta_clean.csv              # Product metadata
│   └── embeddings/
│       ├── text/
│       │   └── combined_1280d_normalized.npy
│       └── image/
│           └── clip_image_768d_normalized.npy
└── ai_fashion_assistant_v1/
    └── data/raw/images/                # Product images (*.jpg)
```

## 📈 Future Enhancements

- [ ] Multi-turn conversational search
- [ ] Query rewriting with LLM (+12% recall)
- [ ] Turkish language fine-tuning
- [ ] Real-time personalization
- [ ] Multi-image outfit search
- [ ] Voice search support

## 📝 Citation

If you use this work, please cite:

```bibtex
@misc{fashion-assistant-2024,
  title={AI Fashion Assistant: Multimodal Search with Deep Learning},
  author={TÜBİTAK 2209-A Project},
  year={2024},
  note={NDCG@10: 86.6\%, Recall@10: 48\%}
}
```

## 📄 License

This project is part of a TÜBİTAK 2209-A research project.

## 🤝 Contributing

This is a research project. For questions or collaboration:
- Open an issue
- Contact via GitHub

## 🎓 Acknowledgments

- **Dataset**: Fashion Product Images (Kaggle)
- **Models**: Hugging Face Transformers
- **Vector Search**: Meta FAISS
- **Framework**: Streamlit

---

**Built with** ❤️ **for the future of fashion e-commerce**

⭐ **Star this repo** if you find it useful!
