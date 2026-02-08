import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score)
import seaborn as sns
import joblib

# Load your data
df = pd.read_csv("data/spam_processed.csv")
X = df["Content"]
y = df["Label"].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# 1. COMPUTE OPTIMAL max_features (TF-IDF vocabulary)

print("\nCOMPUTING optimal max_features")
max_features_range = [1000, 3000, 5000, 7000, 10000]
train_scores1, test_scores1 = validation_curve(
    Pipeline([("tfidf", TfidfVectorizer()), ("xgb", XGBClassifier(n_estimators=100))]),
    X_train, y_train,
    param_name="tfidf__max_features",
    param_range=max_features_range,
    cv=3,
    scoring="accuracy"
)

# Plot TF-IDF learning curve
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(max_features_range, test_scores1.mean(axis=1), 'o-', label="Test Accuracy")
plt.axvline(np.argmax(test_scores1.mean(axis=1)), color='r', linestyle='--', label="Optimal")
optimal_features = max_features_range[np.argmax(test_scores1.mean(axis=1))]
plt.xlabel("max_features")
plt.ylabel("Accuracy")
plt.title("TF-IDF: Optimal Vocabulary Size")
plt.legend()
plt.grid(True)

print(f"Optimal max_features = {optimal_features}")
print(f"Test accuracies: {[f'{s:.3f}' for s in test_scores1.mean(axis=1)]}")

# 2. COMPUTE OPTIMAL n_estimators (XGBoost trees)
# =====================================================
print("\nCOMPUTING optimal n_estimators...")
n_estimators_range = [50, 100, 200, 300, 500]
train_scores2, test_scores2 = validation_curve(
    Pipeline([("tfidf", TfidfVectorizer(max_features=optimal_features)), 
              ("xgb", XGBClassifier())]),
    X_train, y_train,
    param_name="xgb__n_estimators",
    param_range=n_estimators_range,
    cv=3,
    scoring="accuracy"
)

plt.subplot(1, 2, 2)
plt.plot(n_estimators_range, test_scores2.mean(axis=1), 'o-', label="Test Accuracy")
plt.axvline(np.argmax(test_scores2.mean(axis=1)), color='r', linestyle='--', label="Optimal")
optimal_estimators = n_estimators_range[np.argmax(test_scores2.mean(axis=1))]
plt.xlabel("n_estimators")
plt.ylabel("Accuracy")
plt.title("XGBoost: Optimal Number of Trees")
plt.legend()
plt.grid(True)

print(f"Optimal n_estimators = {optimal_estimators}")
print(f"Test accuracies: {[f'{s:.3f}' for s in test_scores2.mean(axis=1)]}")

plt.tight_layout()
plt.savefig("data/optimization_curves.png")
plt.show()

# 3. TRAIN FINAL MODEL with COMPUTED optimal values
# =====================================================
print(f"\nTRAINING FINAL MODEL with optimal parameters:")
print(f"max_features={optimal_features}, n_estimators={optimal_estimators}")

final_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=optimal_features)),
    ("xgb", XGBClassifier(n_estimators=optimal_estimators, random_state=40))
])

final_pipeline.fit(X_train, y_train)
y_pred = final_pipeline.predict(X_test)

print(f"ACCURACY:        {accuracy_score(y_test, y_pred):.3f}")
print(f"PRECISION:       {precision_score(y_test, y_pred):.3f}")
print(f"RECALL:          {recall_score(y_test, y_pred):.3f}")
print(f"F1-SCORE:        {f1_score(y_test, y_pred):.3f}")

joblib.dump(final_pipeline, "model/spam_model.pkl")
