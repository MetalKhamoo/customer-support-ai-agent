import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


# ============================================================
# LOAD DATA
# ============================================================

DATA_FILE = "data/golden/apple_support_golden.csv"

df = pd.read_csv(DATA_FILE)

df = df.dropna(
    subset=["customer_message", "intent"]
)

print("Total examples:", len(df))


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df["customer_message"]
y = df["intent"]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training examples:", len(X_train))
print("Testing examples:", len(X_test))


# ============================================================
# MODEL
# ============================================================

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    )
])


# ============================================================
# TRAIN
# ============================================================

print("\nTraining model...")

model.fit(X_train, y_train)

print("Training complete.")


# ============================================================
# PREDICT
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n======================================")
print(" TF-IDF + Logistic Regression")
print("======================================")

print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)