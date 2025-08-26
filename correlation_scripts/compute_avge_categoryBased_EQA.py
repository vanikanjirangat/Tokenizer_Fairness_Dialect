import pandas as pd
from collections import defaultdict

# Load your CSV file
df = pd.read_csv("metrics_output_Phi_EQA.csv")  # Replace with your actual filename

# Define grouping logic
non_latin = {"arabic", "bengali", "korean"}
high_resource = {"english", "arabic", "korean"}
low_resource = {"swahili", "bengali"}

# Categorize each language entry
def categorize(lang_code):
    lang = lang_code.split("-")[0]
    script = "Non-Latin" if lang in non_latin else "Latin"
    resource = "High" if lang in high_resource else "Low"
    return f"{script}-{resource}"

# Apply categorization
df["category"] = df["lang"].apply(lambda x: categorize(x) if x != "all" else None)
print(df)
# Remove the 'all' row or others without category
df = df.dropna(subset=["category"])

# Compute average metrics by category
grouped = df.groupby("category")[["avg_f1", "avg_em"]].mean().reset_index()

# Print results
print(grouped)

