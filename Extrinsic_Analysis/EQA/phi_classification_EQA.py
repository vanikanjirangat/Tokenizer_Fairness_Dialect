import argparse
import torch
import os
import numpy as np
import pandas as pd
import pickle

from peft import LoraConfig,get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,AutoModelForQuestionAnswering,Trainer,default_data_collator,
)
from datasets import load_dataset, DatasetDict
from trl import SFTTrainer
from transformers.utils import is_flash_attn_2_available
from prompts_QA import get_qa_data_for_ft
import glob
from datasets import load_dataset, Dataset
os.environ["WANDB_PROJECT"] = "Dialect-EQA"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints
lang_code_to_flores_key = {
    "en": "eng_Latn",# reference
    "ru": "rus_Cyrl",
    "fr": "fra_Latn",#French
    "ko": "kor_Hang",
    "ja": "jpn_Jpan",
    "he": "heb_Hebr",
    "hu": "hun_Latn",
    "no": "nob_Latn",
    "hi": "hin_Deva",# Hindi
    "fi": "fin_Latn",
    "es": "spa_Latn",#Spanish
    "de": "deu_Latn",#German
    "it": "ita_Latn",
    "nl": "nld_Latn",
    "zh": "zho_Hans",
    "vi": "vie_Latn",
    "id": "ind_Latn",
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
    "pt": "por_Latn",
    "ht": "hat_Latn",
    "qu": "quy_Latn",
}
def get_specific_layer_names(model):
    # Create a list to store the layer names
    layer_names = []
    
    # Recursively visit all modules and submodules
    for name, module in model.named_modules():
        # Check if the module is an instance of the specified layers
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            # model name parsing 

            layer_names.append('.'.join(name.split('.')[4:]).split('.')[0])
    
    return layer_names

#list(set(get_specific_layer_names(model)))
#tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B", trust_remote_code=True)
#model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B", trust_remote_code=True)
#print(get_specific_layer_names(model))
#print(model)
#print(list(set(get_specific_layer_names(model))))
def flatten_squad_dataset(raw):
    #with open(json_path, "r", encoding="utf-8") as f:
     #   raw = json.load(f)

    flattened = []
    for article in raw["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                for ans in qa["answers"]:
                    flattened.append({
                        "id": qa["id"],
                        "context": context,
                        "question": qa["question"],
                        "answers": {
                            "text": [ans["text"]],
                            "answer_start": [ans["answer_start"]],
                        }
                    })
    return Dataset.from_list(flattened)

def preprocess_qa(examples, tokenizer):
    questions = [q.strip() for q in examples["question"]]
    contexts = examples["context"]
    answers = examples["answers"]

    tokenized = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=512,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized["input_ids"][i]
        #cls_index = input_ids.index(tokenizer.cls_token_id)
        cls_index = 0 # for decoder models
        sequence_ids = tokenized.sequence_ids(i)
        sample_index = sample_mapping[i]

        answer = answers[sample_index]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        if not (start_char >= offsets[token_start_index][0] and end_char <= offsets[token_end_index][1]):
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            start_positions.append(token_start_index - 1)

            while token_end_index >= 0 and offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            end_positions.append(token_end_index + 1)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    return tokenized

def main(args):
    #folders = [f for f in os.listdir("./data/topic_class") if os.path.isdir(f"./data/topic_class/{f}")]
    #folders = ['ind_Latn', 'hau_Latn', 'tuk_Latn', 'ary_Arab', 'tum_Latn', 'cjk_Latn', 'mkd_Cyrl', 'arb_Latn', 'kea_Latn', 'lus_Latn', 'bem_Latn', 'acm_Arab', 'kat_Geor', 'khm_Khmr', 'quy_Latn', 'ewe_Latn', 'pol_Latn', 'snd_Arab', 'dyu_Latn', 'dik_Latn', 'heb_Hebr', 'ceb_Latn', 'fra_Latn', 'bjn_Latn', 'glg_Latn', 'min_Arab', 'taq_Tfng', 'kir_Cyrl', 'ast_Latn', 'hin_Deva', 'kac_Latn', 'tpi_Latn', 'san_Deva', 'ssw_Latn', 'run_Latn', 'lij_Latn', 'fur_Latn', 'npi_Deva', 'hrv_Latn', 'zho_Hant', 'aka_Latn', 'mya_Mymr', 'tel_Telu', 'sot_Latn', 'ces_Latn', 'als_Latn', 'tir_Ethi', 'mai_Deva', 'yue_Hant', 'nld_Latn', 'khk_Cyrl', 'swh_Latn', 'war_Latn', 'cat_Latn', 'ltz_Latn', 'acq_Arab', 'yor_Latn', 'zho_Hans', 'pag_Latn', 'pbt_Arab', 'ukr_Cyrl', 'kan_Knda', 'ckb_Arab', 'lim_Latn', 'crh_Latn', 'sat_Olck', 'lit_Latn', 'tsn_Latn', 'guj_Gujr', 'lao_Laoo', 'dzo_Tibt', 'arz_Arab', 'bak_Cyrl', 'est_Latn', 'arb_Arab', 'luo_Latn', 'grn_Latn', 'mos_Latn', 'hne_Deva', 'plt_Latn', 'bho_Deva', 'hye_Armn', 'tat_Cyrl', 'sin_Sinh', 'kon_Latn', 'gaz_Latn', 'slv_Latn', 'kmr_Latn', 'bos_Latn', 'lvs_Latn', 'taq_Latn', 'awa_Deva', 'isl_Latn', 'ben_Beng', 'nno_Latn', 'vie_Latn', 'kam_Latn', 'bjn_Arab', 'kin_Latn', 'knc_Latn', 'tzm_Tfng', 'nob_Latn', 'shn_Mymr', 'tso_Latn', 'deu_Latn', 'sag_Latn', 'jpn_Jpan', 'srp_Cyrl', 'amh_Ethi', 'bul_Cyrl', 'tgl_Latn', 'uig_Arab', 'ajp_Arab', 'kas_Arab', 'tam_Taml', 'som_Latn', 'kaz_Cyrl', 'swe_Latn', 'ayr_Latn', 'ron_Latn', 'bod_Tibt', 'scn_Latn', 'pes_Arab', 'mar_Deva', 'epo_Latn', 'kik_Latn', 'hat_Latn', 'cym_Latn', 'vec_Latn', 'aeb_Arab', 'ace_Arab', 'szl_Latn', 'lin_Latn', 'bel_Cyrl', 'asm_Beng', 'fuv_Latn', 'gla_Latn', 'tha_Thai', 'kab_Latn', 'sna_Latn', 'eus_Latn', 'hun_Latn', 'ban_Latn', 'sun_Latn', 'fao_Latn', 'oci_Latn', 'pan_Guru', 'uzn_Latn', 'jav_Latn', 'eng_Latn', 'gle_Latn', 'azb_Arab', 'mlt_Latn', 'slk_Latn', 'afr_Latn', 'nya_Latn', 'tgk_Cyrl', 'bam_Latn', 'mri_Latn', 'urd_Arab', 'fij_Latn', 'ars_Arab', 'kas_Deva', 'rus_Cyrl', 'prs_Arab', 'twi_Latn', 'apc_Arab', 'knc_Arab', 'ibo_Latn', 'nus_Latn', 'mal_Mlym', 'zul_Latn', 'xho_Latn', 'kmb_Latn', 'ell_Grek', 'srd_Latn', 'wol_Latn', 'ilo_Latn', 'bug_Latn', 'lug_Latn', 'umb_Latn', 'fin_Latn', 'pap_Latn', 'tur_Latn', 'lua_Latn', 'azj_Latn', 'por_Latn', 'ydd_Hebr', 'mni_Beng', 'smo_Latn', 'ita_Latn', 'ltg_Latn', 'fon_Latn', 'kor_Hang', 'spa_Latn', 'ace_Latn', 'zsm_Latn', 'dan_Latn', 'nso_Latn', 'min_Latn', 'mag_Deva', 'ory_Orya', 'lmo_Latn', 'kbp_Latn']
    #/Question-Answering/SDQA-gold-task
    #folders=list(lang_code_to_flores_key.values())
    #print(f'folders:{len(folders)}')
    qa_dir = "./DialectBench/data/Question-Answering/SDQA-gold-task"
    qa_files = glob.glob(f"{qa_dir}/sdqa-dev-*.json")
    #print(folders)
    for file_path in qa_files:
        lang_code = os.path.basename(file_path).replace("sdqa-dev-", "").replace(".json", "")
        print(f"\nFine-tuning for language: {lang_code}")
        #print(i)
        #dataset = load_dataset("json", data_files={"train": file_path}, field="data")
        #dataset = flatten_squad("Question-Answering/SDQA-gold-task/sdqa-dev-arabic-bhr.json")
        #raw_dataset = load_dataset("json", data_files={"train": file_path})
        #flat_dataset = flatten_squad_dataset(raw_dataset["train"])
        #print(flat_dataset)
        #print(flat_dataset.column_names)
        print(f'Running experiment on dataset {file_path}')
        train_dataset,ds = get_qa_data_for_ft(
            file_path,mode="train")
        #print(f"Sample fraction:{args.train_sample_fraction}")
        print(f"Training samples:{train_dataset.shape}")




        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_ckpt, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
        tokenizer.padding_side = "left"
        #print(f'Running experiment on dataset {folder}')
        #train_dataset, dev_dataset,test_dataset,ds = get_tc_data_for_ft(
         #   folder,mode="train")
        #print(f"Sample fraction:{args.train_sample_fraction}")
        #print(f"Training samples:{train_dataset.shape}")

        #tokenizer = AutoTokenizer.from_pretrained(args.pretrained_ckpt)
        #tokenizer.pad_token = tokenizer.eos_token
    
        print("Getting PEFT method")
    
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            lora_alpha=16,
            r=args.lora_r,
            lora_dropout=args.dropout,
            target_modules=['k_proj', 'v_proj', 'o_proj','q_proj'],#'o_proj', 'qkv_proj', 'gate_up_proj', 'down_proj', 'o_proj', 'qkv_proj', 'gate_up_proj', 'down_proj', 'o_proj', 'qkv_proj', 'gate_up_proj', 'down_proj'],
            )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            )

        model_name_or_path = args.pretrained_ckpt
        print("loading the model")
        torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(
            args.pretrained_ckpt,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map={"": 0},
            #attn_implementation="flash_attention_2",
            )
        print("LOADED!")
        model.config.use_cache = False
        #model = get_peft_model(model, peft_config)

        # Tokenize data
        #tokenized_dataset = flat_dataset.map(
         #   lambda x: preprocess_qa(x, tokenizer),
          #  batched=True,
           # remove_columns=flat_dataset.column_names
        #)
        #print(tokenized_dataset[0])
        lr=2e-5
        results_dir = f"./LLM_experiments/{args.pretrained_ckpt}_classification_epochs-{args.epochs}_rank-{args.lora_r}_dropout-{args.dropout}_lr-{lr}_qa-{lang_code}"# Define training args
    #Best hyperparameters:  {'lr': 0.0005510464326861586, 'batch_size': 4, 'weight_decay': 0.08445563615801001, 'num_train_epochs': 5}
        '''
        training_args = TrainingArguments(
           logging_steps=500,
            report_to="wandb",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=8,
            output_dir=results_dir,
            learning_rate=lr,
            num_train_epochs=num_train_epochs,
            logging_dir=f"{results_dir}/logs",
            fp16=True,
            optim="paged_adamw_32bit",
            lr_scheduler_type="cosine",
            max_grad_norm=0.3,
            save_total_limit=3,
            warmup_ratio=0.03,
            save_strategy="epoch",
            eval_strategy="epoch",  # Ensure to evaluate on validation set
            #eval_steps=500,  # Evaluate every 500 steps
            weight_decay=weight_decay,
            #load_best_model_at_end=True,  # Load best model based on validation
            )

        '''
        check=f"{results_dir}/assets"
        if not os.path.isdir(check):
            training_args = TrainingArguments(
                logging_steps=500,
                report_to="wandb",
                per_device_train_batch_size=2,
                gradient_accumulation_steps=8,
                output_dir=results_dir,
                learning_rate=2e-5,
                num_train_epochs=args.epochs,
                logging_dir=f"{results_dir}/logs",
                fp16=True,
                optim="paged_adamw_32bit",
                lr_scheduler_type="cosine",
                max_grad_norm=0.3,
                warmup_ratio=0.03,
                #weight_decay=0.085,
                )

            print(f"training_args = {training_args}")
            trainer = SFTTrainer(
                model=model,
                args=training_args,
                peft_config=peft_config,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                max_seq_length=512,
                dataset_text_field="instructions",
                packing=True,
                )
            #trainer = Trainer(
             #   model=model,
               # args=training_args,
               # train_dataset=tokenized_dataset,
               # tokenizer=tokenizer,
               # data_collator=default_data_collator,
               # )
            print("FT the mode1")
            torch.cuda.empty_cache()
            trainer_stats = trainer.train()
            train_loss = trainer_stats.training_loss
            print(f"Training loss:{train_loss}")
            print("FT done!")
            peft_model_id = f"{results_dir}/assets"
            trainer.model.save_pretrained(peft_model_id)
            tokenizer.save_pretrained(peft_model_id)

            with open(f"{results_dir}/results.pkl", "wb") as handle:
                run_result = [
                    args.epochs,
                    args.lora_r,
                    args.dropout,
                    train_loss,
                ]
                pickle.dump(run_result, handle)
    #del model
    #del trainer
    #import gc
    #gc.collect()
    #gc.collect()
            torch.cuda.empty_cache()
            print(f'Experiment over in dataset {file_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_ckpt", default="microsoft/Phi-3.5-mini-instruct")
    #model_path="microsoft/Phi-3.5-mini-instruct"
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--max_steps", default=10, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--train_sample_fraction", default=0.99, type=float)
    #parser.add_argument("--dataset_folder", default="")
    args = parser.parse_args()
    main(args)
