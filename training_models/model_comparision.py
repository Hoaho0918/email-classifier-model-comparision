import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           classification_report, confusion_matrix)
import joblib

# Load your data (EXACT same split as both training scripts)
df = pd.read_csv("/workspaces/email-classifier-xgboost/data/spam_processed.csv")
X = df["Content"]
y = df["Label"].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

print("\nLOADING TRAINED MODELS FOR COMPARISON")
print("XGBoost model: /workspaces/email-classifier-xgboost/models/spam_model.pkl")
print("Naive Bayes model: /workspaces/email-classifier-xgboost/models/nb_spam_model.pkl")

# Load both trained pipelines
xgb_model = joblib.load("/workspaces/email-classifier-xgboost/models/xgboost_model.pkl")
nb_model = joblib.load("/workspaces/email-classifier-xgboost/models/NB_model.pkl")

models = {
    "XGBoost": xgb_model,
    "Naive Bayes": nb_model
}

# 1. COMPUTE METRICS FOR SIDE-BY-SIDE COMPARISON
# =====================================================
print("\n1. COMPUTING TEST SET METRICS FOR ALL MODELS")
results = []

for name, model in models.items():
    print(f"\nEvaluating {name}...")
    y_pred = model.predict(X_test)
    
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred)
    })

# Create comparison table
results_df = pd.DataFrame(results).round(3)
print("\nMODEL COMPARISON RESULTS:")
print(results_df.to_string(index=False, float_format='%.3f'))

# Save results
results_df.to_csv("/workspaces/email-classifier-xgboost/training_models/model_comparison_results.csv", index=False)
print("\n✓ Results saved to model_comparison_results.csv")

# 2. PLOT CONFUSION MATRICES SIDE-BY-SIDE
# =====================================================
print("\n2. GENERATING CONFUSION MATRICES")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for i, (name, model) in enumerate(models.items()):
    cm = confusion_matrix(y_test, model.predict(X_test))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i])
    axes[i].set_title(f"{name} Confusion Matrix")
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual")

plt.tight_layout()
plt.savefig("/workspaces/email-classifier-xgboost/training_models/confusion_matrices_comparison.png", dpi=300, bbox_inches='tight')
plt.show()
print("✓ Confusion matrices saved to confusion_matrices_comparison.png")

# 3. PRINT DETAILED CLASSIFICATION REPORTS
# =====================================================
print("\n3. DETAILED CLASSIFICATION REPORTS")
for name, model in models.items():
    print(f"\n{name}:")
    print(classification_report(y_test, model.predict(X_test), 
                              target_names=['ham', 'spam']))

print("\n✓ COMPARISON COMPLETE")
print("\nFiles generated:")
print("  - model_comparison_results.csv")
print("  - confusion_matrices_comparison.png")

