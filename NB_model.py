import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

class Model:
    def __init__(self):
        """Multinomial Naive Bayes text classifier with TF-IDF features."""
        self.vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
        self.encoder = LabelEncoder()
        self.classifier = MultinomialNB()

    def fit(self, X_raw, y_raw):
        """Train model on text and labels."""
        X = self.vectorizer.fit_transform(X_raw)
        y = self.encoder.fit_transform(y_raw)
        self.classifier.fit(X, y)

    def predict(self, X_raw):
        """Predict labels from raw text."""
        X = self.vectorizer.transform(X_raw)
        y_pred = self.classifier.predict(X)
        return self.encoder.inverse_transform(y_pred)
