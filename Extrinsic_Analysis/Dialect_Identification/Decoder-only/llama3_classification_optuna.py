import argparse
import torch
import os
import numpy as np
import pandas as pd
import pickle
import optuna
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
from transformers.utils import is_flash_attn_2_available
from prompts_GDI import get_di_data_for_ft
os.environ["WANDB_PROJECT"] = "Dialect Identifications"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {'accuracy': acc, 'f1': f1}
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
    train_dataset, test_dataset,ds = get_di_data_for_ft(
        mode="train"
        #, train_sample_fraction=args.train_sample_fraction
    )
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
        target_modules=['k_proj', 'v_proj', 'down_proj', 'up_proj', 'o_proj', 'gate_proj', 'q_proj'],#'o_proj', 'qkv_proj', 'gate_up_proj', 'down_proj', 'o_proj', 'qkv_proj', 'gate_up_proj', 'down_proj', 'o_proj', 'qkv_proj', 'gate_up_proj', 'down_proj'],
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
       # attn_implementation="flash_attention_2",
    )
    print("LOADED!")
    model.config.use_cache = False

    # Objective function for Optuna
    def objective(trial):
        lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)
        batch_size = trial.suggest_categorical("batch_size", [2, 4,8])
        weight_decay = trial.suggest_uniform("weight_decay", 0.0, 0.1)
        num_train_epochs = trial.suggest_int("num_train_epochs", 3, 10)
        #results_dir = f"./LLM_experiments/{args.pretrained_ckpt}_classification_epochs-{args.epochs}_rank-{args.lora_r}_dropout-{args.dropout}_GDI_hyperOptuna"
        #results_dir="./LLM_experiments/"
        results_dir = f"./LLM_experiments/checkpoint-{trial.number}"
        os.makedirs(results_dir, exist_ok=True)
        training_args = TrainingArguments(
            logging_steps=500,
            #report_to="wandb",
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

        print(f"training_args = {training_args}")
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            peft_config=peft_config,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            max_seq_length=512,
            dataset_text_field="instructions",
            packing=True,
            #compute_metrics=compute_metrics,
        )
        print("FT the mode1")
        torch.cuda.empty_cache()
        trainer_stats = trainer.train()
        train_loss = trainer_stats.training_loss
        print(f"Training loss:{train_loss}")
        print("FT done!")
        eval_results = trainer.evaluate()
        return eval_results["eval_loss"]  # Minimize eval_loss
    #peft_model_id = f"{results_dir}/assets"
    #trainer.model.save_pretrained(peft_model_id)
    #tokenizer.save_pretrained(peft_model_id)

    #with open(f"{results_dir}/results.pkl", "wb") as handle:
     #   run_result = [
      #      args.epochs,
       #     args.lora_r,
        #    args.dropout,
         #   train_loss,
        #]
        #pickle.dump(run_result, handle)
    #del model
    #del trainer
    #import gc
    #gc.collect()
    #gc.collect()
    #logs = trainer.state.log_history
    #eval_losses = [log["eval_loss"] for log in logs if "eval_loss" in log]

    # Print the eval losses
    #print(eval_losses)
    # Create Optuna study
    study = optuna.create_study(direction="minimize")  # We want to minimize eval_loss
    study.optimize(objective, n_trials=10)  # Run 10 trials

    # Best hyperparameters
    print("Best hyperparameters: ", study.best_params)
    torch.cuda.empty_cache()
    print("Experiment over")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_ckpt", default="meta-llama/Llama-3.2-3B")
    #model_path="microsoft/Phi-3.5-mini-instruct"
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--max_steps", default=10, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--train_sample_fraction", default=0.99, type=float)

    args = parser.parse_args()
    main(args)
