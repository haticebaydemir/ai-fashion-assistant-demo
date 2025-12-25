"""
AI Fashion Assistant - Production Streamlit Demo
=================================================

A professional multimodal fashion search system powered by:
- FAISS vector search
- mpnet + CLIP embeddings  
- Fashion Product Images Dataset (44K products)

Author: AI Fashion Assistant Team
Date: December 2024
"""

import streamlit as st
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import io
import time
from typing import List, Dict, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    """Centralized configuration"""
    
    # Paths (will be auto-detected)
    PROJECT_ROOT = Path("/content/drive/MyDrive/ai_fashion_assistant_v2")
    DATA_DIR = PROJECT_ROOT / "data/processed"
    EMB_DIR = PROJECT_ROOT / "embeddings"
    IMAGES_DIR = Path("/content/drive/MyDrive/ai_fashion_assistant_v1/data/raw/images")
    
    # Embedding dimensions
    TEXT_DIM = 1280  # mpnet 768 + CLIP text 512
    IMAGE_DIM = 768  # CLIP image
    
    # Model names
    TEXT_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    CLIP_MODEL = "openai/clip-vit-base-patch32"
    
    # Search parameters
    DEFAULT_K = 12
    MAX_K = 24
    
    # Performance
    CACHE_TTL = 3600  # 1 hour

config = Config()

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Fashion Assistant",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': """
        # AI Fashion Assistant v2.0
        
        Multimodal fashion search powered by AI.
        
        **Technologies:**
        - mpnet (multilingual semantic understanding)
        - CLIP (vision-language alignment)
        - FAISS (fast similarity search)
        - LightGBM (learning-to-rank)
        
        **Performance:**
        - NDCG@10: 86.6%
        - Recall@10: 48%
        - Latency: <100ms
        
        **Dataset:**
        - Fashion Product Images (Kaggle)
        - 44,446 products
        - 7 categories, 3 genders
        """
    }
)

# ============================================================
# CUSTOM CSS STYLING
# ============================================================

st.markdown("""
<style>
    /* Hero Section */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2.5rem 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        opacity: 0.95;
        margin-bottom: 1rem;
    }
    
    .hero-stats {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-top: 1.5rem;
        flex-wrap: wrap;
    }
    
    .hero-stat {
        background: rgba(255,255,255,0.2);
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        backdrop-filter: blur(10px);
    }
    
    .hero-stat strong {
        font-size: 1.4rem;
        display: block;
    }
    
    .hero-stat span {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Live Status Badge */
    .status-badge {
        background: #10b981;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 1rem;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        background: white;
        border-radius: 50%;
        animation: blink 1.5s infinite;
    }
    
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }
    
    /* Search Box */
    .search-container {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin: 1.5rem 0;
    }
    
    /* Example Queries */
    .example-queries {
        display: flex;
        gap: 0.75rem;
        flex-wrap: wrap;
        margin: 1rem 0;
    }
    
    .example-query {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        cursor: pointer;
        transition: transform 0.2s, box-shadow 0.2s;
        border: none;
    }
    
    .example-query:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Product Cards */
    .product-card {
        background: white;
        border-radius: 12px;
        overflow: hidden;
        transition: transform 0.3s, box-shadow 0.3s;
        border: 1px solid #e5e7eb;
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    
    .product-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.15);
    }
    
    .product-image-container {
        position: relative;
        width: 100%;
        padding-top: 133%; /* 3:4 aspect ratio */
        overflow: hidden;
        background: #f9fafb;
    }
    
    .product-image {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    
    .product-badge {
        position: absolute;
        top: 10px;
        right: 10px;
        background: rgba(16, 185, 129, 0.95);
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        backdrop-filter: blur(10px);
    }
    
    .product-info {
        padding: 1rem;
        flex-grow: 1;
        display: flex;
        flex-direction: column;
    }
    
    .product-name {
        font-size: 1rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.5rem;
        line-height: 1.4;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    
    .product-meta {
        display: flex;
        flex-direction: column;
        gap: 0.4rem;
        margin-top: auto;
    }
    
    .product-category {
        font-size: 0.85rem;
        color: #6b7280;
    }
    
    .product-score {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.9rem;
    }
    
    .score-bar {
        flex-grow: 1;
        height: 6px;
        background: #e5e7eb;
        border-radius: 3px;
        overflow: hidden;
    }
    
    .score-fill {
        height: 100%;
        background: linear-gradient(90deg, #10b981 0%, #059669 100%);
        border-radius: 3px;
        transition: width 0.6s ease-out;
    }
    
    /* Explainability */
    .explain-box {
        background: #f0fdf4;
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 0.5rem;
        font-size: 0.85rem;
    }
    
    .explain-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin: 0.3rem 0;
        color: #065f46;
    }
    
    /* Stats Box */
    .stat-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        display: block;
        margin-bottom: 0.25rem;
    }
    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Quality Indicator */
    .quality-indicator {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .quality-bar {
        height: 30px;
        background: #e5e7eb;
        border-radius: 15px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .quality-fill {
        height: 100%;
        background: linear-gradient(90deg, #10b981 0%, #059669 100%);
        border-radius: 15px;
        transition: width 0.8s ease-out;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 1rem;
        color: white;
        font-weight: 600;
    }
    
    /* Loading State */
    .loading-shimmer {
        background: linear-gradient(90deg, #f3f4f6 25%, #e5e7eb 50%, #f3f4f6 75%);
        background-size: 200% 100%;
        animation: shimmer 1.5s infinite;
    }
    
    @keyframes shimmer {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2rem;
        }
        
        .hero-stats {
            gap: 1rem;
        }
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Improve button styling */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

def init_session_state():
    """Initialize session state variables"""
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    
    if 'current_user' not in st.session_state:
        st.session_state.current_user = "Guest"
    
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = {
            'preferences': [],
            'search_count': 0,
            'clicked_items': [],
            'favorite_categories': []
        }
    
    if 'backend_loaded' not in st.session_state:
        st.session_state.backend_loaded = False
    
    if 'last_results' not in st.session_state:
        st.session_state.last_results = None

init_session_state()

# ============================================================
# BACKEND: REAL SEARCH ENGINE
# ============================================================

@st.cache_resource(show_spinner=False)
def load_backend():
    """
    Load all backend components with progress tracking.
    
    Returns:
        dict: Backend components (embeddings, products, models)
    """
    import faiss
    from sentence_transformers import SentenceTransformer
    from transformers import CLIPProcessor, CLIPModel
    import torch
    
    backend = {}
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backend['device'] = device
    
    try:
        # 1. Load products CSV
        products_path = config.DATA_DIR / "meta_clean.csv"
        if not products_path.exists():
            st.error(f"❌ Products file not found: {products_path}")
            return None
        
        backend['products_df'] = pd.read_csv(products_path)
        
        # 2. Load text embeddings (normalized)
        text_emb_path = config.EMB_DIR / "text" / "combined_1280d_normalized.npy"
        if not text_emb_path.exists():
            st.error(f"❌ Text embeddings not found: {text_emb_path}")
            return None
        
        backend['text_embeddings'] = np.load(text_emb_path).astype('float32')
        
        # 3. Load image embeddings (normalized)
        image_emb_path = config.EMB_DIR / "image" / "clip_image_768d_normalized.npy"
        if not image_emb_path.exists():
            st.error(f"❌ Image embeddings not found: {image_emb_path}")
            return None
        
        backend['image_embeddings'] = np.load(image_emb_path).astype('float32')
        
        # 4. Create FAISS indexes
        # Text index (1280d)
        text_index = faiss.IndexFlatIP(config.TEXT_DIM)
        text_index.add(backend['text_embeddings'])
        backend['text_index'] = text_index
        
        # Image index (768d)
        image_index = faiss.IndexFlatIP(config.IMAGE_DIM)
        image_index.add(backend['image_embeddings'])
        backend['image_index'] = image_index
        
        # 5. Load models (lazy - only if needed)
        backend['text_model'] = None  # Will be loaded on first text search
        backend['clip_model'] = None  # Will be loaded on first image search
        backend['clip_processor'] = None
        
        # 6. Cache model loading functions
        def get_text_model():
            if backend['text_model'] is None:
                backend['text_model'] = SentenceTransformer(config.TEXT_MODEL)
                backend['text_model'].to(device)
            return backend['text_model']
        
        def get_clip_model():
            if backend['clip_model'] is None:
                backend['clip_model'] = CLIPModel.from_pretrained(config.CLIP_MODEL)
                backend['clip_processor'] = CLIPProcessor.from_pretrained(config.CLIP_MODEL)
                backend['clip_model'].to(device)
            return backend['clip_model'], backend['clip_processor']
        
        backend['get_text_model'] = get_text_model
        backend['get_clip_model'] = get_clip_model
        
        return backend
        
    except Exception as e:
        st.error(f"❌ Backend loading failed: {str(e)}")
        return None

class FashionSearchEngine:
    """Production-grade fashion search engine"""
    
    def __init__(self, backend: Dict):
        self.backend = backend
        self.products_df = backend['products_df']
        self.text_index = backend['text_index']
        self.image_index = backend['image_index']
        self.device = backend['device']
    
    def text_search(
        self,
        query: str,
        k: int = 12,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search by text query.
        
        Args:
            query: Search query
            k: Number of results
            filters: Optional filters (category, gender, etc.)
        
        Returns:
            List of product dictionaries with scores
        """
        import torch
        from sklearn.preprocessing import normalize
        
        # Get text model
        text_model = self.backend['get_text_model']()
        
        # Encode query
        query_emb = text_model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        # Normalize
        query_emb = normalize(query_emb, norm='l2').astype('float32')
        
        # Search with FAISS
        k_search = min(k * 3, len(self.products_df))  # Over-fetch for filtering
        distances, indices = self.text_index.search(query_emb, k_search)
        
        # Get results
        results = []
        for i, (idx, score) in enumerate(zip(indices[0], distances[0])):
            if idx >= len(self.products_df):
                continue
            
            product = self.products_df.iloc[idx].to_dict()
            product['score'] = float(score)
            product['rank'] = i + 1
            
            # Apply filters
            if filters:
                if 'categories' in filters and filters['categories']:
                    if product.get('subCategory') not in filters['categories']:
                        continue
                
                if 'colors' in filters and filters['colors']:
                    if product.get('baseColour') not in filters['colors']:
                        continue
                
                if 'gender' in filters and filters['gender']:
                    if product.get('gender') not in filters['gender']:
                        continue
            
            results.append(product)
            
            if len(results) >= k:
                break
        
        return results
    
    def image_search(
        self,
        image: Image.Image,
        k: int = 12,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search by image.
        
        Args:
            image: PIL Image
            k: Number of results
            filters: Optional filters
        
        Returns:
            List of product dictionaries with scores
        """
        import torch
        from sklearn.preprocessing import normalize
        
        # Get CLIP model
        clip_model, clip_processor = self.backend['get_clip_model']()
        
        # Preprocess image
        inputs = clip_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Encode
        with torch.no_grad():
            image_emb = clip_model.get_image_features(**inputs)
            image_emb = image_emb.cpu().numpy()
        
        # Normalize
        image_emb = normalize(image_emb, norm='l2').astype('float32')
        
        # Search with FAISS
        k_search = min(k * 3, len(self.products_df))
        distances, indices = self.image_index.search(image_emb, k_search)
        
        # Get results
        results = []
        for i, (idx, score) in enumerate(zip(indices[0], distances[0])):
            if idx >= len(self.products_df):
                continue
            
            product = self.products_df.iloc[idx].to_dict()
            product['score'] = float(score)
            product['rank'] = i + 1
            
            # Apply filters
            if filters:
                if 'categories' in filters and filters['categories']:
                    if product.get('subCategory') not in filters['categories']:
                        continue
                
                if 'colors' in filters and filters['colors']:
                    if product.get('baseColour') not in filters['colors']:
                        continue
            
            results.append(product)
            
            if len(results) >= k:
                break
        
        return results
    
    def get_product_image_path(self, product_id: int) -> Optional[Path]:
        """Get image path for a product"""
        image_path = config.IMAGES_DIR / f"{product_id}.jpg"
        return image_path if image_path.exists() else None

# ============================================================
# LOAD BACKEND
# ============================================================

def load_backend_with_progress():
    """Load backend with Streamlit progress bar"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("🔄 Loading products...")
    progress_bar.progress(20)
    time.sleep(0.3)
    
    status_text.text("🔄 Loading embeddings...")
    progress_bar.progress(40)
    time.sleep(0.3)
    
    status_text.text("🔄 Creating FAISS indexes...")
    progress_bar.progress(70)
    time.sleep(0.3)
    
    # Load backend
    backend = load_backend()
    
    if backend is None:
        progress_bar.empty()
        status_text.empty()
        return None
    
    status_text.text("✅ Backend ready!")
    progress_bar.progress(100)
    time.sleep(0.5)
    
    progress_bar.empty()
    status_text.empty()
    
    return backend

# Try to load backend
if not st.session_state.backend_loaded:
    with st.spinner("🚀 Initializing AI backend..."):
        backend = load_backend_with_progress()
        
        if backend is not None:
            st.session_state.backend = backend
            st.session_state.search_engine = FashionSearchEngine(backend)
            st.session_state.backend_loaded = True
            st.success("✅ Backend loaded successfully!")
        else:
            st.error("❌ Failed to load backend. Please check file paths.")
            st.stop()
else:
    backend = st.session_state.backend
    search_engine = st.session_state.search_engine

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def explain_result(product: Dict, query: str) -> str:
    """Generate explanation for why this product was returned"""
    explanations = []
    
    # Category match
    if query.lower() in product.get('subCategory', '').lower():
        explanations.append(f"✓ Kategori eşleşmesi: '{product.get('subCategory')}'")
    
    # Color match
    if product.get('baseColour'):
        color_tr = {
            'Black': 'siyah', 'White': 'beyaz', 'Blue': 'mavi',
            'Red': 'kırmızı', 'Green': 'yeşil', 'Yellow': 'sarı'
        }
        color_lower = product['baseColour'].lower()
        for en, tr in color_tr.items():
            if en.lower() in query.lower() or tr in query.lower():
                if en.lower() == color_lower:
                    explanations.append(f"✓ Renk eşleşmesi: '{product['baseColour']}'")
                    break
    
    # High similarity
    score = product.get('score', 0)
    if score > 0.8:
        explanations.append(f"✓ Yüksek benzerlik: %{score*100:.0f}")
    elif score > 0.6:
        explanations.append(f"✓ İyi benzerlik: %{score*100:.0f}")
    
    # Gender match
    if product.get('gender'):
        gender_tr = {'Men': 'erkek', 'Women': 'kadın', 'Boys': 'erkek çocuk', 'Girls': 'kız çocuk'}
        for en, tr in gender_tr.items():
            if en.lower() in query.lower() or tr in query.lower():
                if product['gender'] == en:
                    explanations.append(f"✓ Cinsiyet uyumu: {product['gender']}")
                    break
    
    if not explanations:
        explanations.append(f"✓ Semantik benzerlik: %{score*100:.0f}")
    
    return "<br>".join(explanations)

def get_search_quality_score(results: List[Dict]) -> Tuple[int, str, List[str]]:
    """
    Calculate search quality score and suggestions.
    
    Returns:
        (score, message, suggestions)
    """
    if not results:
        return 0, "❌ Sonuç bulunamadı", ["Farklı kelimeler deneyin", "Filtreleri kaldırın"]
    
    avg_score = np.mean([r['score'] for r in results])
    
    if avg_score > 0.8:
        return 95, "✅ Mükemmel eşleşme!", ["Sonuçlar çok uyumlu"]
    elif avg_score > 0.6:
        return 75, "✅ İyi sonuçlar", ["Bazı ürünler tam eşleşmiyor"]
    elif avg_score > 0.4:
        return 55, "⚠️ Orta kalite", ["Daha spesifik sorgu deneyin", "Filtreleri ayarlayın"]
    else:
        return 30, "⚠️ Düşük eşleşme", ["Farklı kelimeler kullanın", "Filtreleri değiştirin"]

# ============================================================
# HERO SECTION
# ============================================================

st.markdown("""
<div class="hero-section">
    <div class="hero-title">🎨 AI Fashion Assistant</div>
    <div class="hero-subtitle">
        Yapay Zeka ile Akıllı Moda Arama Sistemi
    </div>
    <div style="margin: 1rem 0;">
        <span style="background: rgba(255,255,255,0.3); padding: 0.4rem 0.8rem; border-radius: 15px; font-size: 0.9rem;">
            Text + Image + Semantic Understanding
        </span>
    </div>
    <div class="hero-stats">
        <div class="hero-stat">
            <strong>86.6%</strong>
            <span>NDCG@10</span>
        </div>
        <div class="hero-stat">
            <strong>48%</strong>
            <span>Recall@10</span>
        </div>
        <div class="hero-stat">
            <strong>&lt;100ms</strong>
            <span>Search Time</span>
        </div>
        <div class="hero-stat">
            <strong>44K</strong>
            <span>Products</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Live status badge
st.markdown("""
<div class="status-badge">
    <div class="status-dot"></div>
    LIVE SYSTEM - Real FAISS Search
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("### 👤 Kullanıcı Profili")
    
    # User selection
    user_options = [
        "Guest",
        "User A (Kadın, Casual)",
        "User B (Erkek, Formal)",
        "User C (Kadın, Sporty)",
        "User D (Erkek, Street)"
    ]
    selected_user = st.selectbox("Profil", user_options)
    
    if selected_user != st.session_state.current_user:
        st.session_state.current_user = selected_user
        st.session_state.user_profile = {
            'preferences': [],
            'search_count': 0,
            'clicked_items': [],
            'favorite_categories': []
        }
    
    st.markdown("---")
    
    # User stats
    st.markdown("### 📊 İstatistikler")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Aramalar", st.session_state.user_profile['search_count'])
    with col2:
        st.metric("Tıklamalar", len(st.session_state.user_profile['clicked_items']))
    
    st.markdown("---")
    
    # Filters
    st.markdown("### 🔍 Filtreler")
    
    # Get unique categories from products
    categories = sorted(backend['products_df']['subCategory'].dropna().unique().tolist())
    filter_categories = st.multiselect(
        "Kategori",
        categories,
        default=[],
        help="Belirli kategorilerde ara"
    )
    
    # Get unique colors
    colors = sorted(backend['products_df']['baseColour'].dropna().unique().tolist())
    filter_colors = st.multiselect(
        "Renk",
        colors,
        default=[],
        help="Belirli renklerde ara"
    )
    
    # Gender filter
    genders = sorted(backend['products_df']['gender'].dropna().unique().tolist())
    filter_gender = st.multiselect(
        "Cinsiyet",
        genders,
        default=[],
        help="Belirli cinsiyetler için ara"
    )
    
    st.markdown("---")
    
    # Advanced options
    with st.expander("⚙️ Gelişmiş"):
        num_results = st.slider(
            "Sonuç sayısı",
            min_value=6,
            max_value=config.MAX_K,
            value=config.DEFAULT_K,
            step=6
        )
        
        use_personalization = st.checkbox("Kişiselleştirme", value=False)
        show_explanations = st.checkbox("Açıklamaları göster", value=True)

# ============================================================
# MAIN CONTENT - SEARCH TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(["🔍 Metin Arama", "📸 Görsel Arama", "📊 İstatistikler"])

# ============================================================
# TAB 1: TEXT SEARCH
# ============================================================

with tab1:
    st.markdown("### Aradığınızı tanımlayın")
    
    # Example queries
    st.markdown("💡 **Örnek aramalar** (tıklayın):")
    
    example_queries = [
        "casual summer dress",
        "black Nike shoes",
        "formal shirt for office",
        "rahat spor ayakkabı",
        "mavi elbise"
    ]
    
    # Create clickable example queries
    cols = st.columns(len(example_queries))
    for i, (col, query) in enumerate(zip(cols, example_queries)):
        with col:
            if st.button(f"🔘 {query}", key=f"example_{i}", use_container_width=True):
                st.session_state.example_query = query
    
    # Search input
    col1, col2 = st.columns([5, 1])
    
    with col1:
        # Use example query if clicked
        default_query = st.session_state.get('example_query', '')
        search_query = st.text_input(
            "Arama",
            value=default_query,
            placeholder="Örnek: 'siyah spor ayakkabı', 'casual dress', 'yaz için elbise'",
            label_visibility="collapsed",
            key="text_search_input"
        )
        # Clear example query after use
        if 'example_query' in st.session_state:
            del st.session_state.example_query
    
    with col2:
        search_button = st.button("🔍 Ara", type="primary", use_container_width=True)
    
    # Perform search
    if search_button and search_query:
        # Update stats
        st.session_state.user_profile['search_count'] += 1
        st.session_state.search_history.append({
            'query': search_query,
            'type': 'text',
            'timestamp': pd.Timestamp.now()
        })
        
        # Search with progress
        with st.spinner("🔍 Aranıyor..."):
            start_time = time.time()
            
            # Prepare filters
            filters = {}
            if filter_categories:
                filters['categories'] = filter_categories
            if filter_colors:
                filters['colors'] = filter_colors
            if filter_gender:
                filters['gender'] = filter_gender
            
            # Search
            results = search_engine.text_search(
                query=search_query,
                k=num_results,
                filters=filters if filters else None
            )
            
            search_time = time.time() - start_time
            
            # Store results
            st.session_state.last_results = results
        
        # Display results header
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"### 📦 {len(results)} sonuç bulundu")
        with col2:
            st.metric("⚡ Hız", f"{search_time*1000:.0f}ms")
        with col3:
            if results:
                avg_score = np.mean([r['score'] for r in results])
                st.metric("🎯 Ortalama Skor", f"{avg_score*100:.0f}%")
        
        # Search quality indicator
        if results:
            quality_score, quality_msg, suggestions = get_search_quality_score(results)
            
            st.markdown(f"""
            <div class="quality-indicator">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <strong>🎯 Arama Kalitesi</strong>
                    <span style="color: #059669; font-weight: 600;">{quality_msg}</span>
                </div>
                <div class="quality-bar">
                    <div class="quality-fill" style="width: {quality_score}%;">
                        {quality_score}%
                    </div>
                </div>
                <div style="font-size: 0.85rem; color: #6b7280;">
                    {'<br>'.join([f'💡 {s}' for s in suggestions])}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Display results grid
        if results:
            cols_per_row = 4
            for i in range(0, len(results), cols_per_row):
                cols = st.columns(cols_per_row)
                
                for j, col in enumerate(cols):
                    if i + j < len(results):
                        product = results[i + j]
                        
                        with col:
                            # Get product image
                            image_path = search_engine.get_product_image_path(product['id'])
                            
                            if image_path and image_path.exists():
                                try:
                                    img = Image.open(image_path)
                                    st.image(img, use_container_width=True)
                                except:
                                    st.image("https://via.placeholder.com/300x400?text=No+Image", use_container_width=True)
                            else:
                                st.image("https://via.placeholder.com/300x400?text=No+Image", use_container_width=True)
                            
                            # Product info
                            st.markdown(f"**{product.get('productDisplayName', 'Unknown')}**")
                            
                            # Category and gender
                            category = product.get('subCategory', 'N/A')
                            gender = product.get('gender', 'N/A')
                            st.caption(f"🏷️ {category} • {gender}")
                            
                            # Color
                            if product.get('baseColour'):
                                st.caption(f"🎨 {product['baseColour']}")
                            
                            # Score
                            score = product.get('score', 0)
                            score_pct = int(score * 100)
                            st.markdown(f"""
                            <div class="product-score">
                                <span>⭐</span>
                                <div class="score-bar">
                                    <div class="score-fill" style="width: {score_pct}%"></div>
                                </div>
                                <span style="font-weight: 600;">{score_pct}%</span>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Explanation (if enabled)
                            if show_explanations:
                                explanation = explain_result(product, search_query)
                                st.markdown(f"""
                                <div class="explain-box">
                                    <div style="font-weight: 600; margin-bottom: 0.3rem;">Neden bu ürün?</div>
                                    {explanation}
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Action button
                            if st.button("👁️ Detay", key=f"view_{product['id']}", use_container_width=True):
                                st.session_state.user_profile['clicked_items'].append(product['id'])
                                st.success(f"✅ Ürün #{product['id']} görüntülendi!")
        else:
            st.warning("😔 Sonuç bulunamadı. Lütfen farklı kelimeler deneyin veya filtreleri kaldırın.")

# ============================================================
# TAB 2: IMAGE SEARCH
# ============================================================

with tab2:
    st.markdown("### Görsel ile arama yapın")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("📸 **Bir ürün görseli yükleyin**")
        st.caption("Benzer ürünleri bulalım!")
        
        uploaded_file = st.file_uploader(
            "Görsel seçin",
            type=['jpg', 'jpeg', 'png'],
            help="Maksimum 5MB",
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Yüklenen görsel", use_container_width=True)
            
            # Search button
            if st.button("🔍 Benzerlerini Bul", type="primary", use_container_width=True):
                # Update stats
                st.session_state.user_profile['search_count'] += 1
                st.session_state.search_history.append({
                    'query': 'image_search',
                    'type': 'image',
                    'timestamp': pd.Timestamp.now()
                })
                
                # Search
                with st.spinner("🎨 Görsel analiz ediliyor..."):
                    start_time = time.time()
                    
                    # Prepare filters
                    filters = {}
                    if filter_categories:
                        filters['categories'] = filter_categories
                    if filter_colors:
                        filters['colors'] = filter_colors
                    if filter_gender:
                        filters['gender'] = filter_gender
                    
                    # Search
                    results = search_engine.image_search(
                        image=image,
                        k=num_results,
                        filters=filters if filters else None
                    )
                    
                    search_time = time.time() - start_time
                    
                    # Store results
                    st.session_state.last_results = results
                
                with col2:
                    st.markdown("### 🎯 Benzer Ürünler")
                    st.success(f"✅ {len(results)} benzer ürün bulundu!")
                    st.metric("⚡ Hız", f"{search_time*1000:.0f}ms")
    
    # Display results
    if uploaded_file and st.session_state.last_results:
        st.markdown("---")
        st.markdown("### 📦 Sonuçlar")
        
        results = st.session_state.last_results
        
        # Results grid
        cols_per_row = 4
        for i in range(0, len(results), cols_per_row):
            cols = st.columns(cols_per_row)
            
            for j, col in enumerate(cols):
                if i + j < len(results):
                    product = results[i + j]
                    
                    with col:
                        # Product image
                        image_path = search_engine.get_product_image_path(product['id'])
                        
                        if image_path and image_path.exists():
                            try:
                                img = Image.open(image_path)
                                st.image(img, use_container_width=True)
                            except:
                                st.image("https://via.placeholder.com/300x400?text=No+Image", use_container_width=True)
                        else:
                            st.image("https://via.placeholder.com/300x400?text=No+Image", use_container_width=True)
                        
                        # Product info
                        st.markdown(f"**{product.get('productDisplayName', 'Unknown')}**")
                        st.caption(f"🏷️ {product.get('subCategory', 'N/A')}")
                        
                        # Similarity score
                        score = product.get('score', 0)
                        score_pct = int(score * 100)
                        st.markdown(f"""
                        <div class="product-score">
                            <span>🎯</span>
                            <div class="score-bar">
                                <div class="score-fill" style="width: {score_pct}%"></div>
                            </div>
                            <span style="font-weight: 600;">{score_pct}%</span>
                        </div>
                        """, unsafe_allow_html=True)

# ============================================================
# TAB 3: STATISTICS
# ============================================================

with tab3:
    st.markdown("### 📊 Sistem İstatistikleri")
    
    # System stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stat-box">
            <span class="stat-value">44,446</span>
            <span class="stat-label">Toplam Ürün</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        num_categories = len(backend['products_df']['subCategory'].unique())
        st.markdown(f"""
        <div class="stat-box">
            <span class="stat-value">{num_categories}</span>
            <span class="stat-label">Kategori</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-box">
            <span class="stat-value">86.6%</span>
            <span class="stat-label">NDCG@10</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="stat-box">
            <span class="stat-value">48%</span>
            <span class="stat-label">Recall@10</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Search history
    if st.session_state.search_history:
        st.markdown("### 📜 Arama Geçmişi")
        history_df = pd.DataFrame(st.session_state.search_history)
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
        history_df = history_df.sort_values('timestamp', ascending=False)
        
        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Henüz arama yapmadınız.")
    
    st.markdown("---")
    
    # Dataset distribution
    st.markdown("### 📊 Dataset Dağılımı")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**En Popüler Kategoriler**")
        category_counts = backend['products_df']['subCategory'].value_counts().head(10)
        st.bar_chart(category_counts)
    
    with col2:
        st.markdown("**En Popüler Renkler**")
        color_counts = backend['products_df']['baseColour'].value_counts().head(10)
        st.bar_chart(color_counts)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown("""
<div style="text-align: center; color: #6b7280; padding: 2rem 0;">
    <p style="margin-bottom: 0.5rem;">
        <strong>AI Fashion Assistant v2.0</strong> - TÜBİTAK 2209-A Projesi
    </p>
    <p style="font-size: 0.9rem; margin-bottom: 1rem;">
        Powered by mpnet + CLIP + FAISS + LightGBM
    </p>
    <p style="font-size: 0.85rem;">
        🔍 44,446 Products • ⚡ <100ms Search • 🎯 86.6% NDCG@10
    </p>
</div>
""", unsafe_allow_html=True)
