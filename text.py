!pip install scikit-learn pandas numpy scipy

import pandas as pd
import numpy as np
import re
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack

from google.colab import files
uploaded = files.upload()

df = pd.read_csv("text_samples.csv")  # uploaded file name

print("✅ Dataset Loaded Successfully!")
print(df.head())

class TextAuthenticityDetector:
    def __init__(self):
        self.vectorizer = None
        self.scaler = None
        self.ensemble = None
        self.is_trained = False
        self.training_stats = {}

    def extract_advanced_features(self, text):
        text = str(text)
        words = text.split()
        sentences = text.split('.')

        features = {
            'char_count': len(text),
            'word_count': len(words),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'sentence_count': len([s for s in sentences if s.strip()]),
            'long_words': sum(1 for w in words if len(w) > 8) / max(len(words), 1),
            'unique_words_ratio': len(set(words)) / max(len(words), 1),
            'formal_words': len(re.findall(r'\b(therefore|thus|hence|moreover|furthermore|however|consequently|additionally)\b', text.lower())),
            'technical_terms': len(re.findall(r'\b(probability|distribution|variance|coefficient|hypothesis|statistical|estimation|random|variable|parameter)\b', text.lower())),
            'comma_count': text.count(','),
            'period_count': text.count('.'),
            'colon_count': text.count(':'),
            'punctuation_ratio': sum(1 for c in text if c in '.,;:!?') / max(len(text), 1),
            'has_formula': 1 if re.search(r'[=∑∫σμλ]|P\(|E\(|\^', text) else 0,
            'definition_pattern': len(re.findall(r'\b(is|are|refers to|defined as|known as|represents|describes)\b', text.lower())),
            'question_marks': text.count('?'),
            'numbers': len(re.findall(r'\d+', text)),
            'comprehensive_words': len(re.findall(r'\b(comprehensive|detailed|various|multiple|several|numerous)\b', text.lower())),
            'explanation_words': len(re.findall(r'\b(example|such as|including|for instance|specifically)\b', text.lower())),
        }
        return features

    def train_model(self, df, test_size=0.2, random_state=42):
        """Train ensemble model using 'text' and 'label' columns"""
        texts = df['text'].astype(str).tolist()
        labels = df['label'].map({'real': 0, 'fake': 1}).tolist()

        if len(texts) < 10:
            return False, "❌ Not enough samples (need at least 10)."

        X_train_text, X_test_text, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state, stratify=labels
        )

        self.vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 3), min_df=1, max_df=0.95)
        X_train_tfidf = self.vectorizer.fit_transform(X_train_text)
        X_test_tfidf = self.vectorizer.transform(X_test_text)

      
        train_features = pd.DataFrame([self.extract_advanced_features(t) for t in X_train_text])
        test_features = pd.DataFrame([self.extract_advanced_features(t) for t in X_test_text])

        self.scaler = MinMaxScaler()
        train_scaled = self.scaler.fit_transform(train_features)
        test_scaled = self.scaler.transform(test_features)

        X_train_combined = hstack([X_train_tfidf, train_scaled])
        X_test_combined = hstack([X_test_tfidf, test_scaled])


        nb = MultinomialNB(alpha=0.1)
        lr = LogisticRegression(C=10, max_iter=1000, random_state=42)
        rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

        self.ensemble = VotingClassifier(
            estimators=[('nb', nb), ('lr', lr), ('rf', rf), ('gb', gb)],
            voting='soft', weights=[1, 2, 2, 2]
        )

        self.ensemble.fit(X_train_combined, y_train)

        y_pred = self.ensemble.predict(X_test_combined)
        acc = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(self.ensemble, X_train_combined, y_train, cv=5)

        self.training_stats = {
            'accuracy': acc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'total_samples': len(texts)
        }

        self.is_trained = True
        return True, f"✅ Model trained successfully! Accuracy: {acc:.2%}"

    def predict(self, text):
        if not self.is_trained:
            return "⚠️ Model not trained yet!", 0
        text_tfidf = self.vectorizer.transform([text])
        custom_feats = pd.DataFrame([self.extract_advanced_features(text)])
        scaled = self.scaler.transform(custom_feats)
        text_combined = hstack([text_tfidf, scaled])
        pred = self.ensemble.predict(text_combined)[0]
        prob = self.ensemble.predict_proba(text_combined)[0]
        label = "🤖 AI-Generated" if pred == 1 else "👤 Human-Written"
        confidence = max(prob) * 100
        return label, confidence

detector = TextAuthenticityDetector()
ok, msg = detector.train_model(df)
print(msg)

print("\nTraining Stats:")
print(detector.training_stats)

sample_text = "The new AI model can write essays almost indistinguishable from humans."
label, confidence = detector.predict(sample_text)

print(f"\nPrediction: {label}")
print(f"Confidence: {confidence:.2f}%")
