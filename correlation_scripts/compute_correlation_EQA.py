import pandas as pd
import json

# 1. Load CSV
#csv_df = pd.read_csv("metrics_output1.csv")  # path to scores
csv_df = pd.read_csv("metrics_output_llama_EQA.csv")
#csv_df = pd.read_csv("updated_file_mistral.csv")

#metrics_output_llama3.csv 
# 2. Load JSON
#with open("./Information-Parity/flores_information_parity_results_Phi-Mini.json") as f:  
  #json_data = json.load(f)
with open("./Information-Parity/flores_information_parity_results_Llama3.json") as f:  # IP values
  json_data = json.load(f)
# 3. Language mapping
#lang_code_to_flores_key = {
 #   "en": "eng_Latn",
  #  "ru": "rus_Cyrl",
   # "fr": "fra_Latn",
    #"ko": "kor_Hang",
   # "spa": "spa_Latn",
   # "min": "min_Arab",
    # add more if needed
#}
lang_code_to_flores_key = {
    "en": "eng_Latn",# reference
    "ru": "rus_Cyrl",
    "fr": "fra_Latn",#French
    "ko": "kor_Hang",#Korean
    "ja": "jpn_Jpan",
    "he": "heb_Hebr",
    "hu": "hun_Latn",
    "no": "nob_Latn",
    "hi": "hin_Deva",# Hindi
    "fi": "fin_Latn",#Finnish
    "es": "spa_Latn",#Spanish
    "de": "deu_Latn",#German
    "it": "ita_Latn",
    "nl": "nld_Latn",
    "zh": "zho_Hans",
    "vi": "vie_Latn",
    "id": "ind_Latn",#Indonesian
    "ro": "ron_Latn",
    "uk": "ukr_Cyrl",
    "sr": "srp_Cyrl",
    "hr": "hrv_Latn",
    "da": "dan_Latn",
    "ca": "cat_Latn",
    "ar": "arb_Arab",#Arabic
    "tr": "tur_Latn",
    "cs": "ces_Latn",
    "th": "tha_Thai",
    "bn": "ben_Beng",
    "bg": "bul_Cyrl",
    "el": "ell_Grek",
    "ur": "urd_Arab",
    "mr": "mar_Deva",
    "eu": "eus_Latn",
    "et": "est_Latn",
    "ms": "zsm_Latn",
    "as": "asm_Beng",
    "gu": "guj_Gujr",
    "ka": "kat_Geor",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "np": "npi_Deva",
    "or": "ory_Orya",
    "pa": "pan_Guru",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "my": "mya_Mymr",
    "sw": "swh_Latn",
    "pr": "por_Latn",
    "ht": "hat_Latn",
    "qu": "quy_Latn",
    "aj": "azj_Latn",
    "ck": "ckb_Arab",
    "nl": "nld_Latn",#Dutch
    "iv": "lvs_Latn",
    "lj": "lij_Latn",
    "zh": "zho_Hans",
    "ns": "nso_Latn",
}
# FLORES-200 language map
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
    }
csv_df['main_lang'] = csv_df['lang'].str.split('-').str[0]
csv_df['flores_code'] = csv_df['main_lang'].map(lang_map)
#print(csv_df)

#******
flores_to_lang = {v: k for k, v in lang_code_to_flores_key.items()}

# 4. Add short lang codes to the CSV data
csv_df["lang_orig"] = csv_df["lang"]
csv_df["lang"] = csv_df["flores_code"].map(flores_to_lang)
#csv_df["lang"] = csv_df["lang"].map(flores_to_lang)
#print(csv_df)
#nan_lang_rows = csv_df[csv_df["lang"].isna()]
#print("Rows with NaN in 'lang':")
#print(nan_lang_rows)

unmapped = csv_df[csv_df["lang"].isna()]
print("Unmapped languages from CSV:")
print(unmapped[["lang"]].drop_duplicates())

csv_df = csv_df.dropna(subset=["lang"])
print(csv_df)
# 5. Extract IP metrics from JSON
ip_data = []
for lang, stats in json_data["results"].items():
    ip_data.append({
        "lang": lang,
        "ip_mean": stats["mean"],
        "ip_std": stats["std"],
        "ip_median": stats["median"]
    })
ip_df = pd.DataFrame(ip_data)
#print(ip_df)
# 6. Merge CSV and JSON data on `lang`
final_df = pd.merge(csv_df, ip_df, on="lang", how="inner")
print(final_df)
# 7. Export to CSV
#final_df.to_csv("merged_lang_metrics_Phi_mini.csv", index=False)
#final_df.to_csv("merged_lang_metrics_Phi_EQA.csv", index=False)
print("Merged data saved to 'merged_lang_metrics.csv'")
correlation = final_df["ip_mean"].corr(final_df["avg_f1"])
print(f"Pearson correlation between IP mean and avg_f1: {correlation:.4f}")
cor_macro=final_df["ip_mean"].corr(final_df["avg_em"])
print(f"Pearson correlation between IP mean and avg_em: {correlation:.4f}")
import pandas as pd
from scipy.stats import pearsonr

# Load your DataFrame
# df = pd.read_csv("your_file.csv")  # Assuming your data is in a CSV

# Define high-resource language names
high_resource = {
    'arb_Arab','fra_Latn', 'spa_Latn',  'rus_Cyrl', 'hin_Deva',
    'deu_Latn', 'zho_Hans', 'eng_Latn'
}

# Function to determine category
def classify(row):
    lang_name = row['flores_code']
    lang_code = row['flores_code']
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
df=final_df
# Apply classification
df['Category'] = df.apply(classify, axis=1)

# Function to compute correlation per category
def compute_correlations(df):
    results = []
    for cat in df['Category'].unique():
        subset = df[df['Category'] == cat]
        x = subset['ip_mean']
        #y = subset['eval_f1_macro']
        y=subset['avg_f1']
        if len(subset) >= 1:  # Require at least 3 points to compute correlation
            corr, pval = pearsonr(x, y)
            results.append((cat, corr, pval, len(subset)))
        else:
            results.append((cat, None, None, len(subset)))
    return pd.DataFrame(results, columns=['Category', 'Correlation', 'p-value', 'Sample Size'])

# Compute and display
correlation_results = compute_correlations(df)
print(correlation_results)
