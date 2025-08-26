import argparse
import torch
import os
import numpy as np
import pandas as pd
import pickle

from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
from transformers.utils import is_flash_attn_2_available
from prompts_TC import get_tc_data_for_ft
import os
os.environ["HF_TOKEN"] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["WANDB_PROJECT"] = "Dialect-Topic Classification"  # name your W&B project
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
def main(args):
    folders = [f for f in os.listdir("./data/topic_class") if os.path.isdir(f"./data/topic_class/{f}")]
    folders = ['ind_Latn', 'hau_Latn', 'tuk_Latn', 'ary_Arab', 'tum_Latn', 'cjk_Latn', 'mkd_Cyrl', 'arb_Latn', 'kea_Latn', 'lus_Latn', 'bem_Latn', 'acm_Arab', 'kat_Geor', 'khm_Khmr', 'quy_Latn', 'ewe_Latn', 'pol_Latn', 'snd_Arab', 'dyu_Latn', 'dik_Latn', 'heb_Hebr', 'ceb_Latn', 'fra_Latn', 'bjn_Latn', 'glg_Latn', 'min_Arab', 'taq_Tfng', 'kir_Cyrl', 'ast_Latn', 'hin_Deva', 'kac_Latn', 'tpi_Latn', 'san_Deva', 'ssw_Latn', 'run_Latn', 'lij_Latn', 'fur_Latn', 'npi_Deva', 'hrv_Latn', 'zho_Hant', 'aka_Latn', 'mya_Mymr', 'tel_Telu', 'sot_Latn', 'ces_Latn', 'als_Latn', 'tir_Ethi', 'mai_Deva', 'yue_Hant', 'nld_Latn', 'khk_Cyrl', 'swh_Latn', 'war_Latn', 'cat_Latn', 'ltz_Latn', 'acq_Arab', 'yor_Latn', 'zho_Hans', 'pag_Latn', 'pbt_Arab', 'ukr_Cyrl', 'kan_Knda', 'ckb_Arab', 'lim_Latn', 'crh_Latn', 'sat_Olck', 'lit_Latn', 'tsn_Latn', 'guj_Gujr', 'lao_Laoo', 'dzo_Tibt', 'arz_Arab', 'bak_Cyrl', 'est_Latn', 'arb_Arab', 'luo_Latn', 'grn_Latn', 'mos_Latn', 'hne_Deva', 'plt_Latn', 'bho_Deva', 'hye_Armn', 'tat_Cyrl', 'sin_Sinh', 'kon_Latn', 'gaz_Latn', 'slv_Latn', 'kmr_Latn', 'bos_Latn', 'lvs_Latn', 'taq_Latn', 'awa_Deva', 'isl_Latn', 'ben_Beng', 'nno_Latn', 'vie_Latn', 'kam_Latn', 'bjn_Arab', 'kin_Latn', 'knc_Latn', 'tzm_Tfng', 'nob_Latn', 'shn_Mymr', 'tso_Latn', 'deu_Latn', 'sag_Latn', 'jpn_Jpan', 'srp_Cyrl', 'amh_Ethi', 'bul_Cyrl', 'tgl_Latn', 'uig_Arab', 'ajp_Arab', 'kas_Arab', 'tam_Taml', 'som_Latn', 'kaz_Cyrl', 'swe_Latn', 'ayr_Latn', 'ron_Latn', 'bod_Tibt', 'scn_Latn', 'pes_Arab', 'mar_Deva', 'epo_Latn', 'kik_Latn', 'hat_Latn', 'cym_Latn', 'vec_Latn', 'aeb_Arab', 'ace_Arab', 'szl_Latn', 'lin_Latn', 'bel_Cyrl', 'asm_Beng', 'fuv_Latn', 'gla_Latn', 'tha_Thai', 'kab_Latn', 'sna_Latn', 'eus_Latn', 'hun_Latn', 'ban_Latn', 'sun_Latn', 'fao_Latn', 'oci_Latn', 'pan_Guru', 'uzn_Latn', 'jav_Latn', 'eng_Latn', 'gle_Latn', 'azb_Arab', 'mlt_Latn', 'slk_Latn', 'afr_Latn', 'nya_Latn', 'tgk_Cyrl', 'bam_Latn', 'mri_Latn', 'urd_Arab', 'fij_Latn', 'ars_Arab', 'kas_Deva', 'rus_Cyrl', 'prs_Arab', 'twi_Latn', 'apc_Arab', 'knc_Arab', 'ibo_Latn', 'nus_Latn', 'mal_Mlym', 'zul_Latn', 'xho_Latn', 'kmb_Latn', 'ell_Grek', 'srd_Latn', 'wol_Latn', 'ilo_Latn', 'bug_Latn', 'lug_Latn', 'umb_Latn', 'fin_Latn', 'pap_Latn', 'tur_Latn', 'lua_Latn', 'azj_Latn', 'por_Latn', 'ydd_Hebr', 'mni_Beng', 'smo_Latn', 'ita_Latn', 'ltg_Latn', 'fon_Latn', 'kor_Hang', 'spa_Latn', 'ace_Latn', 'zsm_Latn', 'dan_Latn', 'nso_Latn', 'min_Latn', 'mag_Deva', 'ory_Orya', 'lmo_Latn', 'kbp_Latn']
    
    folders=list(lang_code_to_flores_key.values())
    print(f'folders:{len(folders)}')
    
    print(folders)
    for folder in folders:
        #print(i)
        
        print(f'Running experiment on dataset {folder}')
        train_dataset, dev_dataset,test_dataset,ds = get_tc_data_for_ft(
            folder,mode="train")
        #print(f"Sample fraction:{args.train_sample_fraction}")
        print(f"Training samples:{train_dataset.shape}")

        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_ckpt)
        tokenizer.pad_token = tokenizer.eos_token
    
        print("Getting PEFT method")
    
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            lora_alpha=16,
            r=args.lora_r,
            lora_dropout=args.dropout,
            target_modules=['k_proj', 'v_proj', 'down_proj', 'up_proj', 'o_proj', 'gate_proj', 'q_proj']
            #['k_proj', 'v_proj', 'o_proj','q_proj'],#'o_proj', 'qkv_proj', 'gate_up_proj', 'down_proj', 'o_proj', 'qkv_proj', 'gate_up_proj', 'down_proj', 'o_proj', 'qkv_proj', 'gate_up_proj', 'down_proj'],
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
        lr=2e-5
        results_dir = f"./LLM_experiments/{args.pretrained_ckpt}_classification_epochs-{args.epochs}_rank-{args.lora_r}_dropout-{args.dropout}_lr-{lr}_TC_{folder}"# Define training args
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
        assets_dir=os.path.join(results_dir,"/assets")
        if not os.path.isdir(assets_dir):
            print(f"No assets {folder}")
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
            #model.gradient_checkpointing_enable()
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
            print(f'Experiment over in dataset {folder}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_ckpt", default="meta-llama/Llama-3.2-3B")
    #model_path="microsoft/Phi-3.5-mini-instruct"
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--max_steps", default=10, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--train_sample_fraction", default=0.99, type=float)
    #parser.add_argument("--dataset_folder", default="")
    args = parser.parse_args()
    main(args)
