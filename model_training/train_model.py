# model_training/train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os
import math

# Paths
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'news.csv')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must have 'text' and 'label' columns")
    df = df.dropna(subset=['text','label'])
    return df

def compute_test_size(n_samples, n_classes, frac=0.2):
    """
    Return a tuple (test_size, stratify_flag)
    - test_size: either a float in (0,1) or an integer >=1
    - stratify_flag: either True (use stratify=y) or False (don't stratify)
    This ensures we don't request a test set smaller than number of classes.
    """
    # desired number of test rows (rounded)
    desired = max(1, math.floor(n_samples * frac))
    # ensure at least one sample per class if possible
    if desired >= n_classes and desired < n_samples:
        return desired, True
    # try increasing to n_classes if possible
    if n_classes < n_samples:
        if n_classes < n_samples:
            # use n_classes as test rows if possible
            if n_classes < n_samples:
                return n_classes, True if n_classes < n_samples else False
    # fallback: don't stratify, use an integer 1 (or leave default 0.2 if many samples)
    if n_samples >= 5:
        # safe to use fractional split for medium+ datasets
        return frac, False
    # for very small datasets, use integer 1 (but no stratify)
    return 1, False

def train():
    print("Loading data...")
    df = load_data()
    n = len(df)
    print("Data loaded:", n, "rows")
    if n == 0:
        raise RuntimeError("No data found in data/news.csv")

    X = df['text'].astype(str)
    y = df['label'].astype(str)

    classes = sorted(set(y))
    n_classes = len(classes)
    print(f"Detected {n_classes} classes: {classes}")

    test_size, use_stratify = compute_test_size(n, n_classes, frac=0.2)
    if isinstance(test_size, float):
        print(f"Using fractional test_size={test_size} (no integer conversion)")
    else:
        print(f"Using integer test_size={test_size} rows")

    stratify_arg = y if use_stratify else None
    if use_stratify:
        print("Stratifying the split by label.")
    else:
        print("NOT stratifying the split (dataset too small or fallback).")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify_arg
    )

    print("Training vectorizer...")
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8, ngram_range=(1,2), max_features=20000)
    X_train_tf = vectorizer.fit_transform(X_train)
    X_test_tf = vectorizer.transform(X_test)

    print("Training model...")
    model = PassiveAggressiveClassifier(max_iter=1000, random_state=42)
    model.fit(X_train_tf, y_train)

    print("Evaluating...")
    if X_test_tf.shape[0] > 0:
        preds = model.predict(X_test_tf)
        try:
            acc = accuracy_score(y_test, preds)
            print("Accuracy:", acc)
            print(classification_report(y_test, preds))
            print("Confusion Matrix:")
            print(confusion_matrix(y_test, preds))
        except Exception as e:
            print("Evaluation error:", e)
    else:
        print("Not enough test data to evaluate (test set empty).")

    # Save vectorizer and model
    vector_path = os.path.join(MODEL_DIR, 'vectorizer.pkl')
    model_path = os.path.join(MODEL_DIR, 'finalized_model.pkl')

    with open(vector_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"Saved vectorizer to {vector_path}")

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved model to {model_path}")

if __name__ == "__main__":
    train()
