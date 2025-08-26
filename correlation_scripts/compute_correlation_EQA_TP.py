import pandas as pd
import json

# Step 1: Load the files
#metrics_df = pd.read_csv("metrics_output_llama3.csv")
#metrics_df = pd.read_csv("metrics_output_llama_EQA.csv")
metrics_df = pd.read_csv("updated_file_mistral.csv")
flores_map_df = pd.read_csv("./tokenization-fairness/compute/flores_language_map.csv")
with open("./tokenization-fairness/token_parity_scores.json") as f:
    tokenizer_parity = json.load(f)

print(flores_map_df.columns)
#flores_map_df["Language"] = flores_map_df["Language"].str.strip()
flores_map_df["Language"] = flores_map_df["Language"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
flores_map_df[" FLORES-200 code"] = flores_map_df[" FLORES-200 code"].astype(str).str.strip()
#print(flores_map_df.head())
# Step 2: Create mapping from FLORES code to language name
flores_code_to_lang = dict(zip(flores_map_df[" FLORES-200 code"], flores_map_df["Language"]))
print(flores_code_to_lang)
# Step 3: Add a new column to metrics_df with the language name

#print(tokenizer_parity)

metrics_df["lang"] = metrics_df["lang"].astype(str).str.strip()
print(set(metrics_df["lang"]) - set(flores_code_to_lang.keys()))
lang_map = {
    "arabic":
        "arab_Arab"
    ,
    "english":
        "eng_Latn"
    ,
    # Add other language mappings here
    "swahili":
        "swh_Latn"
    ,

    "bengali":"ben_Beng",
    "korean":"kor_Hang"
    #"standardarabic":"arabic"
    }
mp={"arabic":"standard arabic","english":"english","swahili":"swahili","bengali":"bengali","korean":"korean"}
metrics_df['main_lang'] = metrics_df['lang'].str.split('-').str[0].str.lower()
metrics_df['Language_Name'] = metrics_df['main_lang'].map(lang_map)
metrics_df['main_lang'] = metrics_df['main_lang'].map(mp)
#metrics_df["Language_Name"] = metrics_df["lang"].map(flores_code_to_lang
print(metrics_df.head())

metrics_df = metrics_df.dropna(subset=["main_lang"])
# Step 4: Get tokenizer parity values for LLaMA3 model
#tp_dict = tokenizer_parity["Llama3"]
#tp_dict = {k.strip().lower(): v for k, v in tokenizer_parity["Llama3"].items()}
#print(llama_tp_dict)
tp_dict = {k.strip().lower(): v for k, v in tokenizer_parity["mBERT"].items()}
print(tp_dict)
# Step 5: Map TP to the metrics dataframe
metrics_df["Tokenizer_Parity"] = metrics_df["main_lang"].map(tp_dict)
print(metrics_df)
# Step 6: Drop rows with missing TP values (optional)
metrics_df = metrics_df.dropna(subset=["Tokenizer_Parity"])

# Step 7: Compute correlation between TP and downstream performance
correlation_matrix = metrics_df[["Tokenizer_Parity", "avg_em", "avg_f1"]].corr()
print("Correlation Matrix:")
print(correlation_matrix)
correlation_matrix = metrics_df[["Tokenizer_Parity", "f1_from_dict"]].corr()
print("Correlation Matrix:")
print(correlation_matrix)
#print(metrics_df[metrics_df["Language_Name"].isna()]["lang"].unique())
#print(set(metrics_df["lang"]) - set(flores_code_to_lang.keys()))
#sample_lang = metrics_df.loc[metrics_df["Language_Name"].isna(), "lang"].iloc[0]
#print(f"Sample problematic lang code: '{sample_lang}'")
#print(f"Is in dict keys? {sample_lang in flores_code_to_lang}")
