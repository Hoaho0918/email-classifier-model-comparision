import joblib
import sys
import pandas as pd

# Load both trained pipelines
print("LOADING TRAINED MODELS...")
xgb_model = joblib.load("/workspaces/email-classifier-xgboost/models/xgboost_model.pkl")
nb_model = joblib.load("/workspaces/email-classifier-xgboost//models/NB_model.pkl")

models = {
    "XGBoost": xgb_model,
    "Naive Bayes": nb_model
}

def predict_email(text, model_name):
    """Predict spam/ham for single email text"""
    model = models[model_name]
    pred = model.predict([text])[0]
    proba_spam = model.predict_proba([text])[0][1]  # Probability of spam (class 1)
    label = "SPAM" if pred == 1 else "HAM"
    confidence = max(model.predict_proba([text])[0]) * 100
    return label, proba_spam, confidence

print("\n" + "="*60)
print("EMAIL SPAM CLASSIFIER - MODEL COMPARISON")
print("="*60)
print("Test any email text with both XGBoost and Naive Bayes models")
print("-" * 60)

# Interactive loop
while True:
    print("\nEnter email text (or 'q' to quit, 'c' to clear history):")
    user_input = input("> ").strip()
    
    if user_input.lower() == 'q':
        print("Goodbye!")
        break
    elif user_input.lower() == 'c':
        print("History cleared.")
        continue
    elif not user_input:
        print("Please enter some text.")
        continue
    
    print("\n" + "-"*60)
    print("PREDICTIONS:")
    print("-" * 60)
    
    # Test both models
    results = []
    for model_name in ["XGBoost", "Naive Bayes"]:
        label, spam_prob, confidence = predict_email(user_input, model_name)
        results.append({
            "Model": model_name,
            "Prediction": label,
            "Spam Probability": f"{spam_prob:.3f}",
            "Confidence": f"{confidence:.1f}%"
        })
        print(f"{model_name:12}: {label} | Spam: {spam_prob:.3f} | Conf: {confidence:.1f}%")
    
    # Show agreement/disagreement
    xgb_pred = results[0]["Prediction"]
    nb_pred = results[1]["Prediction"]
    if xgb_pred == nb_pred:
        print(f"\n BOTH MODELS AGREE: {xgb_pred}")
    else:
        print(f"\n MODELS DISAGREE!")
        print(f"   XGBoost: {xgb_pred} | Naive Bayes: {nb_pred}")
    
    # Save to history CSV (optional)
    results_df = pd.DataFrame(results)
    results_df.to_csv("/workspaces/email-classifier-xgboost/prediction/prediction_history.csv", index=False)

print("\nThanks for testing!")