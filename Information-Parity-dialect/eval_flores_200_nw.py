from datasets import load_dataset
from information_parity.information_parity import InformationParity
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import numpy as np
import json
import datetime
import os
from datasets import Dataset
os.environ["HF_TOKEN"] = ''
# Define language mappings
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
#ALL_LANGS=("eng_Latn" "ita_Latn" "azj_Latn" "ckb_Arab" "nob_Latn" "nld_Latn" "lvs_Latn" "arb_Arab" "lij_Latn" "zho_Hans" "spa_Latn" "nso_Latn" "ben_Beng")--NLI
#"arabic" "bengali" "english" "finnish" "indonesian" "korean" "russian" "swahili" "telugu"

# Function to evaluate information parity for a batch of sentences
def evaluate_information_parity(dataset, ip, languages_to_evaluate):
    dict_keys = languages_to_evaluate.keys()
    key_prefix = "sentence_"
    # All languages except English (which is our reference language)
    dict_keys_without_english = [key for key in dict_keys if key != "en"]

    # Format keys for dataset access
    english_key = f"{key_prefix}{languages_to_evaluate['en']}"

    results = {key: [] for key in dict_keys_without_english}

    # Simple loop over all examples in the dataset
    for i in tqdm(range(len(dataset))):
        english_text = dataset[i][english_key]

        for lang_code in dict_keys_without_english:
            lang_key = f"{key_prefix}{languages_to_evaluate[lang_code]}"
            lang_text = dataset[i][lang_key]
            parity_score = ip.compute_pair_information_parity(english_text, lang_text)
            results[lang_code].append(parity_score)

    return results


def load_flores_dataset(flores_path, languages_to_evaluate):
    key_prefix = "sentence_"
    files = sorted(os.listdir(f"{flores_path}/devtest"))
    lang_codes = list(languages_to_evaluate.values())

    # Dictionary to store lists of sentences for each language
    data = {}

    for code in lang_codes:
        #eng_Latn.devtest
        file_path = os.path.join(flores_path, "devtest", f"{code}.devtest")
        with open(file_path, "r", encoding="utf-8") as f:
            data[f"{key_prefix}{code}"] = [line.strip() for line in f]

    # Convert to HuggingFace Dataset for compatibility
    dataset = Dataset.from_dict(data)
    return dataset

def main():
    print("Loading FLORES-200 dataset...")
    flores_path = "./flores200_dataset"  # path to FLORES dev set
    #lang_files = sorted(os.listdir(f"{flores_path}/devtest"))
    #lang_codes = [f.split(".")[0] for f in lang_files if f.endswith(".dev")]
    #dataset = load_dataset("facebook/flores", "all")
    #dev_test = dataset["devtest"]
    
    # Choose which languages to evaluate
    # To evaluate more languages, uncomment and modify the following line:
    #mappings=pd.read_csv("flores_language_map.csv")
    #lang_code_dict = dict(zip(df["FLORES-200 code"],df["language"]))
    #print(len(lang_code_dict))
    #N = 10
    #languages_to_evaluate = dict(list(lang_code_dict.items())[:N])
    languages_to_evaluate = {k: v for k, v in lang_code_to_flores_key.items()}
    print(f'no. of languages evaluated: {len(languages_to_evaluate)}')
    dataset = load_flores_dataset(flores_path, languages_to_evaluate)
    print(dataset)
    print(
        f"Will evaluate the following languages: {list(languages_to_evaluate.keys())}"
    )
    model_dict={"Mixtral":"mistralai/Mixtral-8x7B-Instruct-v0.1","mistral":"mistralai/Mistral-7B-Instruct-v0.2","falcon":"tiiuae/falcon-7b","Phi-mini":"microsoft/Phi-3.5-mini-instruct","Phi-MoE":"microsoft/Phi-3.5-MOE-instruct","Gemma":"google/gemma-7b","Llama":"meta-llama/Llama-3.2-3B","silma":"silma-ai/SILMA-9B-Instruct-v1.0","meltemi":"ilsp/Meltemi-7B-v1.5","bloomz":"bigscience/bloomz","bloom":"bigscience/bloom","nllb":"facebook/nllb-200-distilled-600M"}
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    access_token = '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'  # Add your token if needed
    models=list(model_dict.keys())
    print(models[0])
    for model_name in [models[0]]:
        print(f'Evaluating {model_dict[model_name]}')
        tokenizer = AutoTokenizer.from_pretrained(
                #"meta-llama/Llama-2-7b-chat-hf", 
                model_dict[model_name],token=access_token)
        
        model = AutoModelForCausalLM.from_pretrained(
            #"meta-llama/Llama-2-7b-chat-hf",
            model_dict[model_name],
            token=access_token,
            #device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        # Initialize the InformationParity evaluator
        ip = InformationParity(
            model=model,
            tokenizer=tokenizer,
            is_sentence_piece_tokenizer=True,
        )
        print(ip.is_sentence_piece_tokenizer)

        # Run evaluation
        print("Evaluating information parity across languages...")
        parity_scores = evaluate_information_parity(dataset, ip, languages_to_evaluate)

        # Calculate and display statistics
        dict_keys_without_english = [
            key for key in languages_to_evaluate.keys() if key != "en"
        ]

        print("\nInformation Parity Results (English as reference):")
        print("-" * 50)
        print(f"{'Language':10} | {'Average':10} | {'Std Dev':10} | {'Median':10}")
        print("-" * 50)

        for lang_code in dict_keys_without_english:
            scores = parity_scores[lang_code]
            avg_score = np.mean(scores)
            std_dev = np.std(scores)
            median = np.median(scores)
            print(f"{lang_code:10} | {avg_score:.4f}    | {std_dev:.4f}    | {median:.4f}")

        # Save results to file
        results_data = {
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": f'model_name',
            "results": {
                lang: {
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "median": float(np.median(scores)),
                    "raw_scores": [float(score) for score in scores],
                }
                for lang, scores in parity_scores.items()
            },
        }

        with open(f"flores_information_parity_results_{model_name}.json", "w") as f:
            json.dump(results_data, f, indent=2)

        print(f"Results saved!")


if __name__ == "__main__":
    main()
