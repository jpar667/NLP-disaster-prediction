"""Shared, fixed pipeline for baseline evaluation and final submission."""
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def make_model(metadata):
    transformers = [
        ('word_tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                      max_features=30_000, sublinear_tf=True), 'text'),
        ('char_tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2,
                                      max_features=50_000, sublinear_tf=True), 'text'),
    ]
    for column in metadata:
        encoder = OneHotEncoder(handle_unknown='ignore',
                                min_frequency=5 if column == 'location' else None)
        transformers.append((column, encoder, [column]))
    return Pipeline([
        ('features', ColumnTransformer(transformers, remainder='drop', sparse_threshold=1.0)),
        ('classifier', LogisticRegression(C=1.0, solver='liblinear', max_iter=1000,
                                          random_state=42)),
    ])
