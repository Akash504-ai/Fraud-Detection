# app/model_utils.py
import os
import pickle

# model directory is one level up from this file: project_root/models
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
VECTOR_PATH = os.path.join(MODEL_DIR, 'vectorizer.pkl')
MODEL_PATH = os.path.join(MODEL_DIR, 'finalized_model.pkl')

_cached = {
    'vectorizer': None,
    'model': None
}

def load_vectorizer():
    """Load (and cache) the TF-IDF vectorizer from models/vectorizer.pkl"""
    if _cached['vectorizer'] is None:
        if not os.path.exists(VECTOR_PATH):
            raise FileNotFoundError(f"Vectorizer not found at {VECTOR_PATH}. Run training first.")
        with open(VECTOR_PATH, 'rb') as f:
            _cached['vectorizer'] = pickle.load(f)
    return _cached['vectorizer']

def load_model():
    """Load (and cache) the trained sklearn model from models/finalized_model.pkl"""
    if _cached['model'] is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run training first.")
        with open(MODEL_PATH, 'rb') as f:
            _cached['model'] = pickle.load(f)
    return _cached['model']
