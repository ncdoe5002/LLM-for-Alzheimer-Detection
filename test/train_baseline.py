import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix


DATASET_CSV = "adress_text_dataset.csv"

def run_train_test(model_name, pipeline, X, y):
    print(f"\n============================")
    print(f"Train/Test: {model_name}")
    print(f"============================")

    # Stratified split keeps class balance same in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Control", "Demented"],
            yticklabels=["Control", "Demented"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

def run_cross_validation(model_name, pipeline, X, y):
    print(f"\n============================")
    print(f"Cross-Validation (5-fold): {model_name}")
    print(f"============================")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    print("Accuracy scores:", scores)
    print(f"Mean accuracy: {scores.mean():.4f} ± {scores.std():.4f}")


def main():
    df = pd.read_csv(DATASET_CSV)

    # Basic checks
    df["text"] = df["text"].fillna("")
    X = df["text"].values
    y = df["label"].values

    print("Total samples:", len(df))
    print("Class counts:\n", df["label"].value_counts())

    # --------------------------
    # Baseline 1: Word TF-IDF + Logistic Regression
    # --------------------------
    word_lr = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    # --------------------------
    # Baseline 2: Char 4-grams TF-IDF + Linear SVM
    # --------------------------
    char_svm = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(4, 4), min_df=2)),
        ("clf", LinearSVC())
    ])

    # Run train/test once (easy to understand)
    run_train_test("Word TF-IDF (1–2 grams) + Logistic Regression", word_lr, X, y)
    run_train_test("Char TF-IDF (4-grams) + Linear SVM", char_svm, X, y)

    # Run cross validation (more reliable)
    run_cross_validation("Word TF-IDF (1–2 grams) + Logistic Regression", word_lr, X, y)
    run_cross_validation("Char TF-IDF (4-grams) + Linear SVM", char_svm, X, y)

if __name__ == "__main__":
    main()