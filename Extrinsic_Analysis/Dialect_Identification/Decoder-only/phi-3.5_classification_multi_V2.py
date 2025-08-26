import argparse
import torch
import os
import numpy as np
import pandas as pd
import pickle
from datasets import Dataset, DatasetDict
import torch.nn as nn
import functools
import torch.nn.functional as F
from peft import LoraConfig,get_peft_model
from peft import (
    PeftConfig,
    PeftModel,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    TrainingArguments,Trainer,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from trl import SFTTrainer
from transformers.utils import is_flash_attn_2_available
from prompts_ES import get_di_data_ML_FT
os.environ["HF_TOKEN"] = ' '
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
#tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-MoE-instruct", trust_remote_code=True)
#model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3.5-MoE-instruct", trust_remote_code=True)
#print(get_specific_layer_names(model))
#print(model)
#print(list(set(get_specific_layer_names(model))))
# preprocess dataset with tokenizer
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
# define which metrics to compute for evaluation
def compute_metrics(p):
    predictions, labels = p
    f1_micro = f1_score(labels, predictions > 0, average = 'micro')
    f1_macro = f1_score(labels, predictions > 0, average = 'macro')
    f1_weighted = f1_score(labels, predictions > 0, average = 'weighted')
    return {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }
# create custom trainer class to be able to pass label weights and calculate mutilabel loss
class CustomTrainer1(Trainer):#CAUSAL LM

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #self.label_weights = label_weights
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        
        # forward pass
        outputs = model(**inputs)
        #print(outputs)
        #print(outputs.shape)

        logits = outputs.get("logits")
        logits = logits[:, -1, :]
        #print(logits.shape[1])
        vocab_size=logits.shape[1]
        # Add a classification head to map logits to num_labels
        self.classification_head = nn.Linear(vocab_size, 2).to("cuda")
        logits = self.classification_head(logits)
        #print(logits)
        #print(logits.shape)        
        # compute custom loss
        loss = F.binary_cross_entropy_with_logits(logits, labels.to(torch.float32))
        return (loss, outputs) if return_outputs else loss

class CustomTrainer(Trainer):#SEQ CLS

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        #self.label_weights = label_weights
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # compute custom loss
        loss = F.binary_cross_entropy_with_logits(logits, labels.to(torch.float32))
        return (loss, outputs) if return_outputs else loss

def main(args):
    infer=0
    if infer==0:
        train_dataset, test_dataset,ds = get_di_data_ML_FT(
            mode="train"
        #, train_sample_fraction=args.train_sample_fraction
        )
        #train_dataset, test_dataset,ds=train_dataset[:10], test_dataset[:10],ds[:10]
        print("############")
        ds=DatasetDict(ds)
        print(ds)
        #print(f"Sample fraction:{args.train_sample_fraction}")
        print(f"Training samples:{train_dataset.shape}")
        print(train_dataset)
        print(train_dataset[:1])
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_ckpt)
        tokenizer.pad_token = tokenizer.eos_token
        tokenized_ds = ds.map(functools.partial(tokenize_examples, tokenizer=tokenizer), batched=True)
        tokenized_ds = tokenized_ds.with_format('torch')
        print("Tokenized_ds")
        print(tokenized_ds)
        print(tokenized_ds["train"][:1])
        print("Getting PEFT method")
    
        #peft_config = LoraConfig(
         #   task_type="CAUSAL_LM",
          #  lora_alpha=16,
           # r=args.lora_r,
           # lora_dropout=args.dropout,
           # target_modules=['q_proj','v_proj', 'o_proj', 'k_proj'],#'o_proj', 'qkv_proj', 'gate_up_proj', 'down_proj', 'o_proj', 'qkv_proj', 'gate_up_proj', 'down_proj', 'o_proj', 'qkv_proj', 'gate_up_proj', 'down_proj'],
        #)
    if infer==1:
        train_dataset, test_dataset,ds = get_di_data_ML_FT(mode="inference"
        #, train_sample_fraction=args.train_sample_fraction
        )
        #train_dataset, test_dataset,ds=train_dataset[:10], test_dataset[:10],ds[:10]
        print("############")
        ds=DatasetDict(ds)
        peft_model_id = "~/scratch/LLM_experiments/Multi/meta-llama/Llama-3.2-3B_classification_epochs-3_rank-8_dropout-0.1_custom/checkpoint-8202/"
        config = PeftConfig.from_pretrained(peft_model_id)
        bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        )
        #model = AutoModelForSequenceClassification.from_pretrained(args.pretrained_ckpt,trust_remote_code=True,device_map={"": 0},
        #attn_implementation="flash_attention_2",
        #quantization_config=bnb_config,num_labels=4)

        
        model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model_name_or_path,
        quantization_config=bnb_config,
        trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, peft_model_id)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenized_ds = ds.map(functools.partial(tokenize_examples, tokenizer=tokenizer), batched=True)
        tokenized_ds = tokenized_ds.with_format('torch')
        print("Tokenized_ds")
        print(tokenized_ds)
    '''
    if infer==1:
        train_dataset, test_dataset,ds = get_di_data_ML_FT(mode="inference"
        #, train_sample_fraction=args.train_sample_fraction
        )
        #train_dataset, test_dataset,ds=train_dataset[:10], test_dataset[:10],ds[:10]
        print("############")
        ds=DatasetDict(ds)
        peft_model_id = "~/scratch/LLM_experiments/Multi/meta-llama/Llama-3.2-3B_classification_epochs-3_rank-8_dropout-0.1_custom/custom//checkpoint-8202"

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
        instructions, labels = test_dataset["instructions"], test_dataset["labels"]
        true=[]
        for instruct, label in zip(instructions, labels):
            input_ids = tokenizer(
                instruct, return_tensors="pt", truncation=True).input_ids.cuda()

            with torch.inference_mode():
                outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=20,
                do_sample=True,
                top_p=0.95,
                temperature=1e-3,
                )
                result = tokenizer.batch_decode(
                outputs.detach().cpu().numpy(), skip_special_tokens=True
                )[0]
                result = result[len(instruct) :]
                result=result.split(",")
                result=[x.strip() for x in result]
                result=[int(x) for x in result if len(x)==1]
                if len(result)==4:
                    results.append(result)
                    ctr += 1
                    print(f"Example {ctr} / {len(instructions)}:")
                    print(f"Label:{label}")
                    print(f"Generated:{result}")
                    print("----------------------------------------")
                    true.append(label)
                labels=true
                metrics = {
                        "micro_f1": f1_score(labels, results, average="micro"),
                        "macro_f1": f1_score(labels, results, average="macro"),
                        "precision": precision_score(labels, results, average="micro"),
                        "recall": recall_score(labels, results, average="micro"),
                        "accuracy": accuracy_score(labels, results),
                        }
                print(metrics)



    '''
    if infer!=1:
        
        peft_config = LoraConfig(
            #task_type="CAUSAL_LM",
            task_type="SEQ_CLS",
            lora_alpha=16,
            r=args.lora_r,
            lora_dropout=args.dropout,
            target_modules=['k_proj', 'v_proj', 'down_proj', 'up_proj', 'o_proj', 'gate_proj', 'q_proj'],)
            #'o_proj', 'qkv_proj', 'gate_up_proj', 'down_proj', 'o_proj', 'qkv_proj', 'gate_up_proj', 'down_proj', 'o_proj', 'qkv_proj', 'gate_up_proj', 'down_
        '''
        peft_config = LoraConfig(
            r = args.lora_r, # the dimension of the low-rank matrices
            lora_alpha = 16, # scaling factor for LoRA activations vs pre-trained weight activations
            target_modules = ['k_proj', 'v_proj', 'down_proj', 'up_proj', 'o_proj', 'gate_proj', 'q_proj'],
            lora_dropout = args.dropout, # dropout probability of the LoRA layers
            #bias = 'none', # wether to train bias weights, set to 'none' for attention layers
            task_type = 'SEQ_CLS'
            )
        '''
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,)

        model_name_or_path = args.pretrained_ckpt
        print("loading the model")
        torch.cuda.empty_cache()
        '''
        model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_ckpt,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map={"": 0},
        #attn_implementation="flash_attention_2",
        #problem_type="multi_label_classification",
         )
        '''
        
        model = AutoModelForSequenceClassification.from_pretrained(args.pretrained_ckpt,trust_remote_code=True,device_map={"": 0},
        #attn_implementation="flash_attention_2",
        quantization_config=bnb_config,num_labels=2)
        
        print("LOADED!")
        model.config.use_cache = False

        model = get_peft_model(model, peft_config)
        model.config.pad_token_id = tokenizer.pad_token_id
        #results_dir = f"~/scratch/LLM_experiments/Multi/{args.pretrained_ckpt}_classification_epochs-{args.epochs}_rank-{args.lora_r}_dropout-{args.dropout}_custom_causal"
    results_dir =f"./LLM_experiments/{args.pretrained_ckpt}_classification_epochs-{args.epochs}_rank-{args.lora_r}_dropout-{args.dropout}_ES_seq"
    # Define training args
    
    training_args = TrainingArguments(
        logging_steps=100,
        #report_to="none",
        per_device_train_batch_size=1,
        per_device_eval_batch_size =1,
        gradient_accumulation_steps=8,
        output_dir=results_dir,
        learning_rate=5e-5,
        num_train_epochs=args.epochs,
        logging_dir=f"{results_dir}/logs",
        fp16=False,
        evaluation_strategy = 'epoch',
        optim="paged_adamw_32bit",  
        lr_scheduler_type="cosine",
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        save_strategy = 'epoch',
        load_best_model_at_end = True,
        report_to="wandb",
    )
    '''
    training_args = TrainingArguments(
    learning_rate=5e-5,  # Higher learning rate for better gradient adjustment
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,  # Effective batch size = 8
    output_dir=results_dir,
    logging_dir=f"{results_dir}/logs",
    num_train_epochs=3,
    logging_steps=50,  # More frequent logging
    evaluation_strategy="steps",  # More frequent evaluation
    eval_steps=500,
    save_strategy="steps",
    load_best_model_at_end=True,
    warmup_ratio=0.01,  # Reduce warm-up if overfitting
    max_grad_norm=1.0,  # Higher gradient norm for better gradient flow
    report_to="wandb",
)
'''
    print(f"training_args = {training_args}")
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        #peft_config=peft_config,
        tokenizer=tokenizer,
        train_dataset=tokenized_ds['train'],
        eval_dataset = tokenized_ds['test'],
        data_collator = functools.partial(collate_fn, tokenizer=tokenizer),
        #max_seq_length=512,
        compute_metrics = compute_metrics,
        #dataset_text_field="instructions",
        #packing=True,
    )
    print("FT the mode1")
    torch.cuda.empty_cache()
    if infer!=1:
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
        import gc
        gc.collect()
        gc.collect()
        torch.cuda.empty_cache()
        #del model
        #del trainer
        predictions = trainer.predict(tokenized_ds['test'])

        # Get the logits from the predictions
        logits = predictions.predictions

        # Apply sigmoid to logits (multilabel classification)
        sigmoid_logits = torch.sigmoid(torch.tensor(logits))
        print("sigmoid_logits",sigmoid_logits)
        # Convert sigmoid outputs to binary (multilabel predictions)
        predicted_labels = (sigmoid_logits > 0.5).int()
        #print(predicted_labels)
        #print(trainer.predict())
        print("Predicted Labels (multilabel):", predicted_labels)
        #true_labels=predictions.label
        true_labels=torch.tensor(tokenized_ds['test']['labels'])
        report = classification_report(true_labels, predicted_labels, target_names=['ES-AR','ES-ES'], zero_division=0)
        #print(report)
        #report = classification_report(true_labels, predicted_labels, target_names=['FR-BE','FR-CA','FR-CH','FR-FR'],zero_division=0)
        print(report)
        print(f1_score(y_true=true_labels, y_pred=predicted_labels, average='weighted'))
    if infer==1:
        print(trainer.evaluate())
        predictions = trainer.predict(tokenized_ds['test'])

        # Get the logits from the predictions
        logits = predictions.predictions

        # Apply sigmoid to logits (multilabel classification)
        sigmoid_logits = torch.sigmoid(torch.tensor(logits))

        # Convert sigmoid outputs to binary (multilabel predictions)
        predicted_labels = (sigmoid_logits > 0.5).int()
        #print(predicted_labels)
        #print(trainer.predict())
        print("Predicted Labels (multilabel):", predicted_labels)
        #true_labels=predictions.label
        true_labels=torch.tensor(tokenized_ds['test']['label'])
        #report = classification_report(true_labels, predicted_labels, target_names=['ES-AR','ES-ES'], zero_division=0)
        #print(report)
        report = classification_report(true_labels, predicted_labels, target_names=['FR-BE','FR-CA','FR-CH','FR-FR'],zero_division=0)
        print(report)
        print(f1_score(y_true=true_labels, y_pred=predicted_labels, average='weighted'))
    import gc
    gc.collect()
    gc.collect()
    torch.cuda.empty_cache()
        
    print("Experiment over")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #meta-llama/Llama-3.2-3B
    #['k_proj', 'v_proj', 'down_proj', 'up_proj', 'o_proj', 'gate_proj', 'q_proj']
    parser.add_argument("--pretrained_ckpt", default="meta-llama/Llama-3.2-3B")
    #model_path="microsoft/Phi-3.5-mini-instruct"
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--max_steps", default=10, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--train_sample_fraction", default=0.99, type=float)

    args = parser.parse_args()
    main(args)
