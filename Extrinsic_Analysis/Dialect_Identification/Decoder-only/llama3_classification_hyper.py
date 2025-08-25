import argparse
import torch
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
from transformers.utils import is_flash_attn_2_available
from prompts_GDI import get_di_data_for_ft
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {'accuracy': acc, 'f1': f1}

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
def tokenize_examples(examples, tokenizer):
    tokenized_inputs = tokenizer(examples['instructions'])
    tokenized_inputs['labels'] = examples['labels']
    return tokenized_inputs
# define custom batch preprocessor
def collate_fn(batch, tokenizer):
    dict_keys = ['input_ids', 'attention_mask', 'labels']
    d = {k: [dic[k] for dic in batch] for k in dict_keys}
    d['input_ids'] = torch.nn.utils.rnn.pad_sequence(
        d['input_ids'], batch_first=True, padding_value=tokenizer.pad_token_id
    )
    d['attention_mask'] = torch.nn.utils.rnn.pad_sequence(
        d['attention_mask'], batch_first=True, padding_value=0
    )
    d['labels'] = torch.stack(d['labels'])
    return d

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
    # Check if the pad_token_id is correctly set
    #print(f"Pad Token: {tokenizer.pad_token}, Pad Token ID: {tokenizer.pad_token_id}")
    # If needed, set the pad_token_id explicitly
    #if tokenizer.pad_token_id is None:
     #   tokenizer.pad_token_id = tokenizer.eos_token_id 
    print("Getting PEFT method")
    #meta-llama/Llama-3.2-3B
    #['k_proj', 'v_proj', 'down_proj', 'up_proj', 'o_proj', 'gate_proj', 'q_proj']
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
        attn_implementation="flash_attention_2",
    
    )
    '''
    
    model = AutoModelForSequenceClassification.from_pretrained(
        args.pretrained_ckpt,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map={"": 0},
        attn_implementation="flash_attention_2",num_labels=4 )
    '''
    print("LOADED!")
    model.config.use_cache = False
    model.config.pad_token_id = model.config.eos_token_id
    results_dir = f"~/scratch/LLM_experiments/{args.pretrained_ckpt}_classification_epochs-{args.epochs}_rank-{args.lora_r}_dropout-{args.dropout}_lr-{args.lr}_gdi_hyp_optuna"# Define training args
    '''
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
    )
    '''
    training_args = TrainingArguments(
        logging_steps=500,
        report_to="wandb",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=4,
        output_dir=results_dir,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_dir=f"{results_dir}/logs",
        fp16=True,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        weight_decay=0.098,  # Add weight decay for regularization
        save_steps=1000,
        save_total_limit=3,  # Limit how many checkpoints to save, for efficient disk usage
        save_strategy="epoch",
        eval_strategy="epoch",  # Ensure to evaluate on validation set
        #eval_steps=500,  # Evaluate every 500 steps
        #load_best_model_at_end=True,  # Load best model based on validation
    )
        #from transformers import EarlyStoppingCallback

        # Define the early stopping callback with patience and threshold
       # early_stopping_callback = EarlyStoppingCallback(
        #early_stopping_patience=3,  # Stop after 3 epochs with no improvement
        #early_stopping_threshold=0.01  # Require at least 1% improvement in validation loss
        #)
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
        #callbacks=[early_stopping_callback],  # Add early stopping to callbacks,
        #compute_metrics=compute_metrics  # Add metrics computation
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
    # Accessing the logs
    logs = trainer.state.log_history
    train_losses = [log["loss"] for log in logs if "loss" in log]

    # Plot training loss
    plt.plot(train_losses)
    plt.xlabel("Training Steps")
    plt.ylabel("Train Loss")
    plt.title("Training Loss During Training")
    plt.show()
    # Extract the evaluation loss
    eval_losses = [log["eval_loss"] for log in logs if "eval_loss" in log]

    # Print the eval losses
    print(eval_losses)
    
    import matplotlib.pyplot as plt

    # Create a plot for evaluation loss
    plt.plot(eval_losses)
    plt.xlabel("Evaluation Steps")
    plt.ylabel("Eval Loss")
    plt.title("Evaluation Loss During Training")
    plt.show()
    #del model
    #del trainer
    import gc
    gc.collect()
    gc.collect()
    #trainer.evaluate()
    #predictions=trainer.predict(test_dataset)
    # Access the predictions and metrics
    #preds = predictions.predictions.argmax(-1)  # Get the predicted class indices
    #labels = predictions.label_ids
    #report = classification_report(true_labels, predicted_labels)
    #print(report)
    #metrics = predictions.metrics  # Evaluation metric  
    #print("Metrics:", metrics)
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
    parser.add_argument("--lr", default=0.00028, type=float)
    parser.add_argument("--train_sample_fraction", default=0.99, type=float)

    args = parser.parse_args()
    main(args)
