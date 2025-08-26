import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
os.environ["HF_TOKEN"] = ''
# Define log loss calculator
class ConditionalEntropyRanker:
    @staticmethod
    def _calculate_log_loss_from_logits(logits, relevant_tokens):
        probs = logits.softmax(dim=-1)
        actual_probs = probs[range(len(relevant_tokens)), relevant_tokens]
        return -actual_probs.log2()

@torch.no_grad()
def compute_log_loss(model, tokenizer, text, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    logits = model(**inputs).logits
    input_ids = inputs["input_ids"][0]
    relevant_tokens = input_ids[1:]  # skip BOS
    return ConditionalEntropyRanker._calculate_log_loss_from_logits(logits[0], relevant_tokens).mean().item()

def compute_log_losses_for_models(flores_path, model_dict, device):
    lang_files = sorted(os.listdir(f"{flores_path}/dev"))
    lang_codes = [f.split(".")[0] for f in lang_files if f.endswith(".dev")]

    df = pd.DataFrame(index=lang_codes)

    for model_name, model_path in model_dict.items():
        print(f"\nLoading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        model.eval()

        log_losses = []
        for lang in tqdm(lang_codes, desc=f"Processing {model_name}"):
            file_path = os.path.join(flores_path, "dev", f"{lang}.dev")
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()[:100]  # limit to 100 examples

            losses = []
            for line in lines:
                try:
                    loss = compute_log_loss(model, tokenizer, line, device)
                    losses.append(loss)
                except Exception as e:
                    print(f"Skipped line due to error: {e}")
            mean_loss = sum(losses) / len(losses) if losses else float("nan")
            log_losses.append(mean_loss)

        df[model_name] = log_losses
        print(log_losses)
    df.to_csv("log_losses_per_language.csv")
    print("\nSaved log losses to: log_losses_per_language.csv")
    return df

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
flores_path = "../flores200_dataset"  # path to FLORES dev set

model_dict={"Mixtral":"mistralai/Mixtral-8x7B-Instruct-v0.1","mistral":"mistralai/Mistral-7B-Instruct-v0.2","falcon":"tiiuae/falcon-7b","Phi-mini":"microsoft/Phi-3.5-mini-instruct","Phi-MoE":"microsoft/Phi-3.5-MOE-instruct","Gemma":"google/gemma-7b","Llama":"meta-llama/Llama-3.2-3B","silma":"silma-ai/SILMA-9B-Instruct-v1.0","meltemi":"ilsp/Meltemi-7B-v1.5","bloomz":"bigscience/bloomz","bloom":"bigscience/bloom","nllb":"facebook/nllb-200-distilled-600M"}
model_dict={"Llama2 7B":"meta-llama/Llama-2-7b-hf"}
#model_dict = {
 #   "Phi_Mini": "microsoft/phi-2",
  #  "Gemma": "google/gemma-2b",
   # "TinyLlama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#}

compute_log_losses_for_models(flores_path, model_dict, device)

