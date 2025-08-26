import json
import csv
import pandas as pd
from glob import glob
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
# Load FLORES map: e.g., "Japanese" → "jpn_Jpan"
flores_map = pd.read_csv("./tokenization-fairness/compute/flores_language_map.csv", header=None, names=["language", "flores_code"])
lang_to_flores = dict(zip(flores_map["language"], flores_map["flores_code"].str.strip()))
del lang_to_flores["Language"]
# Load Tokenization Parity (TP) data
with open("./tokenization-fairness/token_parity_scores.json", "r") as f:
    token_parity = json.load(f)

# Collect all IP files
ip_files = glob("./Information-Parity/flores_information_parity_results_*.json")

rows = []
# Reverse lang_code_to_flores_key for matching TP languages to FLORES
reversed_flores = {v: k for k, v in lang_code_to_flores_key.items()}
print(reversed_flores)
print(lang_to_flores)
for ip_file in ip_files:
    model_name = ip_file.split("_")[-1].replace(".json", "")
    print(model_name)
    # Load Information Parity (IP) results
    with open(ip_file, "r") as f:
        ip_data = json.load(f)
    #print(ip_data)
    #'results': {'ru':
    # Reverse lang_code_to_flores_key for matching TP languages to FLORES
    #reversed_flores = {v: k for k, v in lang_code_to_flores_key.items()}
    #print(reversed_flores)
    for language, flores_code in lang_to_flores.items():
        #print((language,flores_code))
        flores_code=flores_code.strip()
        # Skip if not present in IP or TP
        if flores_code in reversed_flores.keys():
            #print(flores_code)
            ip_flores=reversed_flores[flores_code]
            print(ip_flores)
            if ip_flores not in ip_data['results']:
                
                #print("Hmm")
                continue

            ip_info = ip_data['results'][ip_flores]
            ip_mean = ip_info["mean"]
            ip_std = ip_info["std"]
            ip_median = ip_info["median"]

            # Try to get TP for this language
            print((language,model_name))
            tp_value = token_parity.get(model_name, {}).get(language)
            print((ip_mean,tp_value))
            rows.append({
                "language": language,
                "flores_code": flores_code,
                "ip_flores":ip_flores,
                "model": model_name,
                "ip_mean": ip_mean,
                "ip_std": ip_std,
                "ip_median": ip_median,
                "tp": tp_value
            })

# Save to CSV
df = pd.DataFrame(rows)
df.to_csv("merged_ip_tp_metrics.csv", index=False)
print("Saved to merged_ip_tp_metrics.csv")

