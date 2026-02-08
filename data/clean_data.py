import pandas as pd
import re
from pathlib import Path

# Load the raw dataset
df = pd.read_csv("data/spam.csv", encoding="latin-1")

# Rename the two columns: v1 = spam/ham labels, v2 = content of message
df = df[["v1", "v2"]].rename(columns={"v1": "Label", "v2": "Content"})

# Lowercase Title-case words or mixed case, keep FULL UPPERCASE words unchanged
def filter_case(text):
    def replace(match):
        word = match.group(0)
        if not word.isupper() and not word.islower():
            return word.lower()
        elif word.isupper():
            return word
        else:
            return word
    return re.sub(r'\b[A-Z][a-zA-Z]*\b', replace, text)

df["Content"] = (df["Content"]
    .astype(str)
    .str.strip()
    .str.replace(r'[!?.]{2,}', ' PUNCT ', regex=True)    # Multiple punctuation → PUNCT
    .str.replace(r'[!?.]', '', regex=True)               # Single punctuation → remove
    .str.replace(r'[^\w\s]', ' ', regex=True)             # Remove ALL other special chars
    .str.replace(r'[_\-]', ' ', regex=True)              # Replace - and _ → space
    .str.replace(r'\s+', ' ', regex=True)
    .apply(filter_case)                                  # Filter case
)

# Clean the labels
df["Label"] = df["Label"].astype(str).str.strip().str.lower()

# Remove any rows where Label or Content is empty/null
df = df.dropna(subset=["Label", "Content"])  

# Save the cleaned dataset to a new CSV file
# Create output folder + save
Path("data").mkdir(exist_ok=True)
df.to_csv("data/spam_processed.csv", index=False)
