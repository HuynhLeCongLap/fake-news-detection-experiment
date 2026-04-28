import pandas as pd
from sklearn.model_selection import train_test_split

from preprocessing import preprocess
from features import build_tfidf
from models import train_lr, train_nb, train_rf

def fix_label(x):
    x = str(x).strip().lower()
    if x in ['fake', '0']:
        return 0
    elif x in ['real', '1']:
        return 1
    return 0

def main():
    df = pd.read_csv('data/fake_news_dataset.csv', engine='python', on_bad_lines='skip')

    df['text'] = df['text'].fillna("")
    df['label'] = df['label'].apply(fix_label)

    print("Preprocessing...")
    df['processed_text'] = df['text'].apply(preprocess)

    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], df['label'],
        test_size=0.2, random_state=42
    )

    X_train_tfidf, X_test_tfidf, _ = build_tfidf(X_train, X_test)

    print("Training models...")
    acc_lr = train_lr(X_train_tfidf, y_train, X_test_tfidf, y_test)
    acc_nb = train_nb(X_train_tfidf, y_train, X_test_tfidf, y_test)
    acc_rf = train_rf(X_train_tfidf, y_train, X_test_tfidf, y_test)

    print("\n=== RESULTS ===")
    print(f"Logistic Regression: {acc_lr}")
    print(f"Naive Bayes:        {acc_nb}")
    print(f"Random Forest:      {acc_rf}")

if __name__ == "__main__":
    main()
