from sklearn.feature_extraction.text import TfidfVectorizer

def build_tfidf(X_train, X_test, max_features=5000):
    tfidf = TfidfVectorizer(max_features=max_features)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    return X_train_tfidf, X_test_tfidf, tfidf
