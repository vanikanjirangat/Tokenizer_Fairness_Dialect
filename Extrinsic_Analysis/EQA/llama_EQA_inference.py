import argparse
import os
import pandas as pd
import evaluate
import pickle
import torch
import warnings
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from peft import (
    PeftConfig,
    PeftModel,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
import re
import string
from collections import Counter
from prompts_QA import get_qa_data_for_ft
import glob
metric = evaluate.load("rouge")
warnings.filterwarnings("ignore")


def normalize_text(s):
    """Lowercase, remove punctuation/articles/extra whitespace."""
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    s = ' '.join(s.split())
    return s

def f1_score(prediction, ground_truth):
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)

def exact_match(prediction, ground_truth):
    return normalize_text(prediction) == normalize_text(ground_truth)

def main(args):
    #folders= [f for f in os.listdir("./LLM_experiments/QA/microsoft") if os.path.isdir(f"./LLM_experiments/QA/microsoft//{f}")]
    langs= ['ind_Latn', 'hau_Latn', 'tuk_Latn', 'ary_Arab', 'tum_Latn', 'cjk_Latn', 'mkd_Cyrl', 'arb_Latn', 'kea_Latn', 'lus_Latn', 'bem_Latn', 'acm_Arab', 'kat_Geor', 'khm_Khmr', 'quy_Latn', 'ewe_Latn', 'pol_Latn', 'snd_Arab', 'dyu_Latn', 'dik_Latn', 'heb_Hebr', 'ceb_Latn', 'fra_Latn', 'bjn_Latn', 'glg_Latn', 'min_Arab', 'taq_Tfng', 'kir_Cyrl', 'ast_Latn', 'hin_Deva', 'kac_Latn', 'tpi_Latn', 'san_Deva', 'ssw_Latn', 'run_Latn', 'lij_Latn', 'fur_Latn', 'npi_Deva', 'hrv_Latn', 'zho_Hant', 'aka_Latn', 'mya_Mymr', 'tel_Telu', 'sot_Latn', 'ces_Latn', 'als_Latn', 'tir_Ethi', 'mai_Deva', 'yue_Hant', 'nld_Latn', 'khk_Cyrl', 'swh_Latn', 'war_Latn', 'cat_Latn', 'ltz_Latn', 'acq_Arab', 'yor_Latn', 'zho_Hans', 'pag_Latn', 'pbt_Arab', 'ukr_Cyrl', 'kan_Knda', 'ckb_Arab', 'lim_Latn', 'crh_Latn', 'sat_Olck', 'lit_Latn', 'tsn_Latn', 'guj_Gujr', 'lao_Laoo', 'dzo_Tibt', 'arz_Arab', 'bak_Cyrl', 'est_Latn', 'arb_Arab', 'luo_Latn', 'grn_Latn', 'mos_Latn', 'hne_Deva', 'plt_Latn', 'bho_Deva', 'hye_Armn', 'tat_Cyrl', 'sin_Sinh', 'kon_Latn', 'gaz_Latn', 'slv_Latn', 'kmr_Latn', 'bos_Latn', 'lvs_Latn', 'taq_Latn', 'awa_Deva', 'isl_Latn', 'ben_Beng', 'nno_Latn', 'vie_Latn', 'kam_Latn', 'bjn_Arab', 'kin_Latn', 'knc_Latn', 'tzm_Tfng', 'nob_Latn', 'shn_Mymr', 'tso_Latn', 'deu_Latn', 'sag_Latn', 'jpn_Jpan', 'srp_Cyrl', 'amh_Ethi', 'bul_Cyrl', 'tgl_Latn', 'uig_Arab', 'ajp_Arab', 'kas_Arab', 'tam_Taml', 'som_Latn', 'kaz_Cyrl', 'swe_Latn', 'ayr_Latn', 'ron_Latn', 'bod_Tibt', 'scn_Latn', 'pes_Arab', 'mar_Deva', 'epo_Latn', 'kik_Latn', 'hat_Latn', 'cym_Latn', 'vec_Latn', 'aeb_Arab', 'ace_Arab', 'szl_Latn', 'lin_Latn', 'bel_Cyrl', 'asm_Beng', 'fuv_Latn', 'gla_Latn', 'tha_Thai', 'kab_Latn', 'sna_Latn', 'eus_Latn', 'hun_Latn', 'ban_Latn', 'sun_Latn', 'fao_Latn', 'oci_Latn', 'pan_Guru', 'uzn_Latn', 'jav_Latn', 'eng_Latn', 'gle_Latn', 'azb_Arab', 'mlt_Latn', 'slk_Latn', 'afr_Latn', 'nya_Latn', 'tgk_Cyrl', 'bam_Latn', 'mri_Latn', 'urd_Arab', 'fij_Latn', 'ars_Arab', 'kas_Deva', 'rus_Cyrl', 'prs_Arab', 'twi_Latn', 'apc_Arab', 'knc_Arab', 'ibo_Latn', 'nus_Latn', 'mal_Mlym', 'zul_Latn', 'xho_Latn', 'kmb_Latn', 'ell_Grek', 'srd_Latn', 'wol_Latn', 'ilo_Latn', 'bug_Latn', 'lug_Latn', 'umb_Latn', 'fin_Latn', 'pap_Latn', 'tur_Latn', 'lua_Latn', 'azj_Latn', 'por_Latn', 'ydd_Hebr', 'mni_Beng', 'smo_Latn', 'ita_Latn', 'ltg_Latn', 'fon_Latn', 'kor_Hang', 'spa_Latn', 'ace_Latn', 'zsm_Latn', 'dan_Latn', 'nso_Latn', 'min_Latn', 'mag_Deva', 'ory_Orya', 'lmo_Latn', 'kbp_Latn']
    #print(langs[0:10])
    metrics_list=[]
    #print(folders)
    qa_dir = "./DialectBench/data/Question-Answering/SDQA-gold-task"
    qa_files = glob.glob(f"{qa_dir}/sdqa-test-*.json")
    #print(folders)
    for file_path in qa_files:
        lang_code = os.path.basename(file_path).replace("sdqa-test-", "").replace(".json", "")
        print(f"\nFine-tuning for language: {lang_code}")
        #print(i)
        #dataset = load_dataset("json", data_files={"train": file_path}, field="data")
        #dataset = flatten_squad("Question-Answering/SDQA-gold-task/sdqa-dev-arabic-bhr.json")
        #raw_dataset = load_dataset("json", data_files={"train": file_path})
        #flat_dataset = flatten_squad_dataset(raw_dataset["train"])
        #print(flat_dataset)
        #print(flat_dataset.column_names)
        print(f'Running experiment on dataset {file_path}')
        test_dataset,ds = get_qa_data_for_ft(
            file_path,mode="inference")
        #print(f"Sample fraction:{args.train_sample_fraction}")
        print(f"Test samples:{test_dataset.shape}")
    #for experiment in folders:
        #lang=experiment.split("TC_")[1].split("_")[0]
        experiment= f"./LLM_experiments/QA_nw/meta-llama/Llama-3.2-3B-Instruct_classification_epochs-5_rank-8_dropout-0.1_lr-2e-05_qa-{lang_code}"
        #base_path = "./LLM_experiments/QA/microsoft/"
        #experiment = os.path.join(base_path, experiment)
        experiment = os.path.abspath(experiment)
        print(f'experiment:{experiment}')
        lang = experiment.split('qa_')[-1]
        print(f'Inference on the dataset {lang}')
        #test_dataset,ds = get_tc_data_for_ft(lang,mode="inference")
        #test_dataset=test_dataset[1000:]
        #experiment = args.experiment
        #experiment= f"./LLM_experiments/{args.pretrained_ckpt}_classification_epochs-{args.epochs}_rank-{args.lora_r}_dropout-{args.dropout}_lr-{lr}_TC_{folder}"
        peft_model_id = f"{experiment}/assets"
        config = PeftConfig.from_pretrained(peft_model_id)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            quantization_config=bnb_config,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, peft_model_id)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token

        ctr = 0
        results = []
        results1=[]
        instructions,labels = test_dataset["instructions"], test_dataset["labels"]

        for instruct, label in zip(instructions, labels):
            input_ids = tokenizer(
                instruct, return_tensors="pt", truncation=True
            ).input_ids.cuda()

            with torch.inference_mode():
                outputs = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=20,
                    do_sample=False,#changed to deterministic
                    num_beams=1,
                    top_p=1.0,#full probability distribution
                    temperature=0.0,#ensures no randomness
                )
                result = tokenizer.batch_decode(
                    outputs.detach().cpu().numpy(), skip_special_tokens=True
                )[0]
                result = result[len(instruct) :]
                results.append(result)
                #if ":" in result:
                #result1 = result.split(":")[-1].strip()
                result1 = result.strip().split("\n")[0]
                results1.append(result1)
                ctr += 1
                #print(f"Example {ctr} / {len(instructions)}:")
                #print(f"Label:{label}")
                #print(f"Generated:{result}")
                #print(f"Generated1:{result1}")
                #print("----------------------------------------")
        print(results1[:5])
        f1s = [f1_score(p, r) for p, r in zip(results1, labels)]
        ems = [exact_match(p, r) for p, r in zip(results1, labels)]

        avg_f1 = sum(f1s) / len(f1s)
        avg_em = sum(ems) / len(ems)

        print(f"F1 Score: {avg_f1:.2f}")
        print(f"Exact Match: {avg_em * 100:.2f}%")        
        #metrics = {
         #   "micro_f1": f1_score(labels, results1, average="micro"),
          #  "macro_f1": f1_score(labels, results1, average="macro"),
           # "precision": precision_score(labels, results1, average="micro"),
            #"recall": recall_score(labels, results1, average="micro"),
            #"accuracy": accuracy_score(labels, results1),
        #}
        #print(metrics)
        #print(list(set(labels)))
        #print(list(set(results)))
        #la=list(set(labels))
        #true,pred=[],[]

    #for i,k in enumerate(results):
     #   if k in la:
      #      pred.append(k)
       #     true.append(labels[i])
    #print(list(set(true)))
    #print(list(set(pred)))
        #true=labels
        #print(classification_report(true,results1))
    #print(classification_report(true,results,labels=['BHO', 'HIN', 'AWA', 'MAG', 'BRA']))
        #print(confusion_matrix(true,results1))
        #save_dir = os.path.join(f"{args.experiment}", "metrics")
        #if not os.path.exists(save_dir):
         #   os.makedirs(save_dir)

        #with open(os.path.join(save_dir, "metrics.pkl"), "wb") as handle:
         #   pickle.dump(metrics, handle)
        
        
        print(f"Completed experiment {peft_model_id}")
        print("----------------------------------------")
        metrics={"avg_f1":avg_f1,"avg_em":avg_em}
        metrics["lang"] = lang_code
        metrics_list.append(metrics)

    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(metrics_list)

    # Save the DataFrame to a CSV file
    output_csv_path = "./metrics_output_llama_Instruct_EQA.csv"
    df.to_csv(output_csv_path, index=False)

    print(f"Metrics saved to {output_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="1-8-0.1")
    parser.add_argument("--pretrained_ckpt", default="microsoft/Phi-3.5-mini-instruct")
    #model_path="microsoft/Phi-3.5-mini-instruct"
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--max_steps", default=10, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    #parser.add_argument("--train_sample_fraction", default=0.99, type=float)
    args = parser.parse_args()

    main(args)
