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
from prompts_ILI import get_di_data_for_ft
os.environ["HF_TOKEN"] = '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'
os.environ["WANDB_PROJECT"] = "Dialect Identifications"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints
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
#tokenizer = AutoTokenizer.from_pretrained("silma-ai/SILMA-9B-Instruct-v1.0", trust_remote_code=True)
#model = AutoModelForCausalLM.from_pretrained("silma-ai/SILMA-9B-Instruct-v1.0", trust_remote_code=True)
#print(get_specific_layer_names(model))
#print(model)
#print(list(set(get_specific_layer_names(model))))
def main(args):
    train_dataset, test_dataset = get_di_data_for_ft(
        mode="train"
        #, train_sample_fraction=args.train_sample_fraction
    )
    #print(f"Sample fraction:{args.train_sample_fraction}")
    print(f"Training samples:{train_dataset.shape}")

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_ckpt)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    print("Getting PEFT method")
    
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        lora_alpha=16,
        r=args.lora_r,
        lora_dropout=args.dropout,
        target_modules=['q_proj', 'o_proj', 'down_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj'],#'o_proj', 'qkv_proj', 'gate_up_proj', 'down_proj', 'o_proj', 'qkv_proj', 'gate_up_proj', 'down_proj', 'o_proj', 'qkv_proj', 'gate_up_proj', 'down_proj'],
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
        #device_map={"": 0},
        device_map="auto",
        attn_implementation="eager",
    )
    experiment = args.experiment
    #checkpoint_path=f"{experiment}/checkpoint-1650"
    #checkpoint_path="~/scratch/LLM_experiments/silma-ai/SILMA-9B-Instruct-v1.0_classification_epochs-10_rank-8_dropout-0.1_silma/checkpoint-1650"
    #model=AutoModelForCausalLM.from_pretrained(checkpoint_path)
    print("LOADED!")
    model.config.use_cache = False
    #silma-ai/SILMA-9B-Instruct-v1.0
    
    results_dir = f"~/scratch/LLM_experiments/{args.pretrained_ckpt}_classification_epochs-{args.epochs}_rank-{args.lora_r}_dropout-{args.dropout}_silmaILI"#results_dir = f"experiments/{args.pretrained_ckpt}_classification_epochs-{args.epochs}_rank-{args.lora_r}_dropout-{args.dropout}"
    '''
    # Define training args
    training_args = TrainingArguments(
        logging_steps=100,
        report_to="none",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        output_dir=results_dir,
        learning_rate=2e-4,
        num_train_epochs=args.epochs,
        logging_dir=f"{results_dir}/logs",
        fp16=True,
        optim="paged_adamw_32bit",
        lr_scheduler_type="constant",
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        save_strategy="epoch",
    )
    '''
    training_args = TrainingArguments(
    output_dir=results_dir,                   # Directory to save results
    report_to="wandb"
    evaluation_strategy="epoch",               # Evaluate at the end of each epoch
    learning_rate=5e-5,                        # A moderate learning rate
    per_device_train_batch_size=1,            # Smaller batch size due to model size
    per_device_eval_batch_size=1,             # Smaller batch size for evaluation
    num_train_epochs=5,                        # Number of epochs to train
    weight_decay=0.01,                         # Weight decay for regularization
    logging_dir="./logs",                      # Directory for logging
    logging_steps=100,                         # Log every 100 steps
    fp16=True,                                 # Use mixed precision for efficiency
    load_best_model_at_end=True,               # Load the best model after training
    metric_for_best_model="eval_loss",                # Metric for determining best model
    greater_is_better=True,                    # Whether higher metric values are better
    gradient_accumulation_steps=8,             # Gradient accumulation for effective batch size
    max_grad_norm=1.0,                         # Gradient clipping for stability
    save_total_limit=3,                        # Limit total number of checkpoint saves
    save_strategy="epoch",                     # Save model at the end of every epoch
    fp16_opt_level="O1",                       # Use 'O1' for mixed precision optimization
    warmup_steps=500,                          # Number of warmup steps for learning rate
    lr_scheduler_type="cosine",                # Learning rate scheduler type
    eval_accumulation_steps=1,                 # Accumulate evaluation results
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
    )
    print("FT the mode1")
    torch.cuda.empty_cache()
    import time
    start = time.perf_counter()
    #trainer_stats = trainer.train(resume_from_checkpoint=checkpoint_path)
    trainer_stats=trainer.train()
    train_loss = trainer_stats.training_loss
    print(f"Training loss:{train_loss}")
    print("FT done!")
    peft_model_id = f"{results_dir}/assets"
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)
    end = time.perf_counter()
    elapsed_time = end - start
    print(f"Elapsed time: {elapsed_time} seconds")
    with open(f"{results_dir}/results.pkl", "wb") as handle:
        run_result = [
            args.epochs,
            args.lora_r,
            args.dropout,
            train_loss,
        ]
        pickle.dump(run_result, handle)
    trainer.evaluate()
    # del model
   # del trainer
    import gc
    gc.collect()
    gc.collect()
    torch.cuda.empty_cache()
    print("Experiment over")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_ckpt", default="silma-ai/SILMA-9B-Instruct-v1.0")
    #model_path="microsoft/Phi-3.5-mini-instruct"
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--max_steps", default=10, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--train_sample_fraction", default=0.99, type=float)
    #parser.add_argument("--experiment", default="~/scratch/LLM_experiments/silma-ai/SILMA-9B-Instruct-v1.0_classification_epochs-10_rank-8_dropout-0.1_silma/")
    parser.add_argument("--experiment", default="1-8-0.1")
    args = parser.parse_args()
    main(args)
