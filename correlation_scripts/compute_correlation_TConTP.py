import pandas as pd
import json

# Step 1: Load the files
metrics_df = pd.read_csv("metrics_output_llama3.csv")
metrics_df = pd.read_csv("metrics_output1.csv")
#metrics_df = pd.read_csv("evaluation_mBERT_TC.csv")
flores_map_df = pd.read_csv("./tokenization-fairness/compute/flores_language_map.csv")
with open("./tokenization-fairness/token_parity_scores.json") as f:
    tokenizer_parity = json.load(f)

print(flores_map_df.columns)
#flores_map_df["Language"] = flores_map_df["Language"].str.strip()
flores_map_df["Language"] = flores_map_df["Language"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
flores_map_df[" FLORES-200 code"] = flores_map_df[" FLORES-200 code"].astype(str).str.strip()
print(flores_map_df.head())
# Step 2: Create mapping from FLORES code to language name
flores_code_to_lang = dict(zip(flores_map_df[" FLORES-200 code"], flores_map_df["Language"]))
print(flores_code_to_lang)
# Step 3: Add a new column to metrics_df with the language name
metrics_df["lang"] = metrics_df["lang"].astype(str).str.strip()
print(set(metrics_df["lang"]) - set(flores_code_to_lang.keys()))

metrics_df["Language_Name"] = metrics_df["lang"].map(flores_code_to_lang)
#print(metrics_df.head())
# Step 4: Get tokenizer parity values for LLaMA3 model
#llama_tp_dict = tokenizer_parity["Llama3"]
#llama_tp_dict = {k.strip(): v for k, v in tokenizer_parity["Llama3"].items()}
llama_tp_dict = {k.strip(): v for k, v in tokenizer_parity["Phi-Mini"].items()}
#llama_tp_dict = {k.strip(): v for k, v in tokenizer_parity["mBERT"].items()}
# Step 5: Map TP to the metrics dataframe
metrics_df["Tokenizer_Parity"] = metrics_df["Language_Name"].map(llama_tp_dict)
#print(metrics_df)
# Step 6: Drop rows with missing TP values (optional)
metrics_df = metrics_df.dropna(subset=["Tokenizer_Parity"])
#print(metrics_df)
#md=metrics_df[["Language_Name","lang","eval_f1_macro","Tokenizer_Parity"]]
md=metrics_df[["Language_Name","lang","macro_f1","Tokenizer_Parity"]]
print(md)
# Step 7: Compute correlation between TP and downstream performance

#correlation_matrix = metrics_df[["Tokenizer_Parity", "micro_f1", "macro_f1", "precision", "recall", "accuracy"]].corr()
#eval_f1_macro
#correlation_matrix = metrics_df[["Tokenizer_Parity", "eval_f1_macro"]].corr()
correlation_matrix = metrics_df[["Tokenizer_Parity", "macro_f1"]].corr()
print("Correlation Matrix:")
print(correlation_matrix)


import pandas as pd
from scipy.stats import pearsonr

# Load your DataFrame
# df = pd.read_csv("your_file.csv")  # Assuming your data is in a CSV

# Define high-resource language names
high_resource = {
    'Standard Arabic', 'French', 'Spanish', 'Russian', 'Hindi',
    'German', 'Chinese (Simplified)', 'English'
}

# Function to determine category
def classify(row):
    lang_name = row['Language_Name']
    lang_code = row['lang']
    is_latin = 'Latn' in lang_code
    is_high = lang_name in high_resource
    if is_latin and is_high:
        return 'Latin-High'
    elif is_latin and not is_high:
        return 'Latin-Low'
    elif not is_latin and is_high:
        return 'Non-Latin-High'
    else:
        return 'Non-Latin-Low'
df=md
# Apply classification
df['Category'] = df.apply(classify, axis=1)

# Function to compute correlation per category
def compute_correlations(df):
    results = []
    for cat in df['Category'].unique():
        subset = df[df['Category'] == cat]
        x = subset['Tokenizer_Parity']
        #y = subset['eval_f1_macro']
        y=subset['macro_f1']
        if len(subset) >= 3:  # Require at least 3 points to compute correlation
            corr, pval = pearsonr(x, y)
            results.append((cat, corr, pval, len(subset)))
        else:
            results.append((cat, None, None, len(subset)))
    return pd.DataFrame(results, columns=['Category', 'Correlation', 'p-value', 'Sample Size'])

# Compute and display
correlation_results = compute_correlations(df)
print(correlation_results)






#print(metrics_df[metrics_df["Language_Name"].isna()]["lang"].unique())
#print(set(metrics_df["lang"]) - set(flores_code_to_lang.keys()))
#sample_lang = metrics_df.loc[metrics_df["Language_Name"].isna(), "lang"].iloc[0]
#print(f"Sample problematic lang code: '{sample_lang}'")
#print(f"Is in dict keys? {sample_lang in flores_code_to_lang}")
