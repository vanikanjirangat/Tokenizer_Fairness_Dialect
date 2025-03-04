import argparse
import os
import pandas as pd
import evaluate
import pickle
import torch
import warnings
from datasets import Dataset, DatasetDict
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import torch.nn as nn
import functools
import torch.nn.functional as F
from peft import (
    PeftConfig,
    PeftModel,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,AutoModelForSequenceClassification,Trainer,TrainingArguments,)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
from prompts_GDI import get_di_data_for_ft

metric = evaluate.load("rouge")
warnings.filterwarnings("ignore")
def tokenize_examples(examples, tokenizer):
    tokenized_inputs = tokenizer(examples['instructions'],padding='max_length',  # or set to 'longest' if you want maximum flexibility in padding
        truncation=True,
        max_length=512)
    tokenized_inputs['labels'] = label_encoder.fit_transform(examples['labels'])
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

def main(args):
    _, test_dataset,ds = get_di_data_for_ft(mode="inference")
    #test_dataset=test_dataset[1000:]
    ds=DatasetDict(ds)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_ds = ds.map(functools.partial(tokenize_examples, tokenizer=tokenizer), batched=True)
    tokenized_ds = tokenized_ds.with_format('torch')
    experiment = args.experiment

    peft_model_id = f"{experiment}/assets/"
    config = PeftConfig.from_pretrained(peft_model_id)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model_name_or_path,
        quantization_config=bnb_config,
        trust_remote_code=True,num_labels=4,
    )
    model = PeftModel.from_pretrained(model, peft_model_id)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    ctr = 0
    results = []
    instructions, labels = test_dataset["instructions"], test_dataset["labels"]
    '''
    for instruct, label in zip(instructions, labels):
        input_ids = tokenizer(
            instruct, return_tensors="pt", truncation=True
        ).input_ids.cuda()

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
            results.append(result)
            ctr += 1
            print(f"Example {ctr} / {len(instructions)}:")
            print(f"Label:{label}")
            print(f"Generated:{result}")
            print("----------------------------------------")

    metrics = {
        "micro_f1": f1_score(labels, results, average="micro"),
        "macro_f1": f1_score(labels, results, average="macro"),
        "precision": precision_score(labels, results, average="micro"),
        "recall": recall_score(labels, results, average="micro"),
        "accuracy": accuracy_score(labels, results),
    }
    print(metrics)
    print(list(set(labels)))
    print(list(set(results)))
    la=list(set(labels))
    #true,pred=[],[]

    #for i,k in enumerate(results):
     #   if k in la:
      #      pred.append(k)
       #     true.append(labels[i])
    #print(list(set(true)))
    #print(list(set(pred)))
    true=labels
    print(classification_report(true,results))
    #print(classification_report(true,results,labels=['BHO', 'HIN', 'AWA', 'MAG', 'BRA']))
    print(confusion_matrix(true,results))
    save_dir = os.path.join(f"{args.experiment}", "metrics")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, "metrics.pkl"), "wb") as handle:
        pickle.dump(metrics, handle)
    '''
    tokenized_ds = tokenized_ds.map(
    lambda x: {'input_ids': x['input_ids'], 'attention_mask': x['attention_mask'], 'labels': x['labels']},remove_columns=tokenized_ds['test'].column_names)  # Remove all original columns)
    print(tokenized_ds["test"])
    print(tokenized_ds["test"][0])
    #data_collator = functools.partial(collate_fn, tokenizer=tokenizer)
    t=tokenized_ds['test']
    #model.eval()
    #results_dir=f"./LLM_experiments/{args.pretrained_ckpt}_classification_epochs-{args.epochs}_rank-{args.lora_r}_dropout-{args.dropout}_gdiseq"
    results_dir="./"
    training_args = TrainingArguments(
        logging_steps=500,
        report_to="wandb",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=1,
        output_dir=results_dir,
        learning_rate=5e-5,
        num_train_epochs=10,
        logging_dir=f"{results_dir}/logs",
        fp16=False,
        optim="paged_adamw_32bit",
        lr_scheduler_type="linear",
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        weight_decay=0.01,  # Add weight decay for regularization
        save_total_limit=2,  # Limit how many checkpoints to save, for efficient disk usage
        evaluation_strategy="steps",  # Ensure to evaluate on validation set
        eval_steps=500,  # Evaluate every 500 steps
        load_best_model_at_end=True,  # Load best model based on validation
    )
    from transformers import EarlyStoppingCallback

    # Define the early stopping callback with patience and threshold
    early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=3,  # Stop after 3 epochs with no improvement
    early_stopping_threshold=0.01  # Require at least 1% improvement in validation loss
    )
    print(f"training_args = {training_args}")
    trainer =Trainer(
        model=model,
        args=training_args,
        #peft_config=peft_config,
        #tokenizer=tokenizer,
        train_dataset=tokenized_ds['train'],
        eval_dataset=tokenized_ds['test'],
        #data_collator=data_collator,
        data_collator = functools.partial(collate_fn, tokenizer=tokenizer),
        #max_seq_length=512,
        #dataset_text_field="instructions",
        #packing=True,
        callbacks=[early_stopping_callback],  # Add early stopping to callbacks,
        #compute_metrics=compute_metrics  # Add metrics computation
    )

    #predictions=model.predict(t)
    #labels=
    #predictions = torch.argmax(predictions, dim=-1).cpu().numpy()
    
    predictions=trainer.predict(t)
    # Access the predictions and metrics
    predicted_labels = predictions.predictions.argmax(-1)  # Get the predicted class indices
    true_labels = predictions.label_ids
    print((true_labels[:10],predicted_labels[:10]))
    report = classification_report(true_labels, predicted_labels)
    print(report)
    #metrics = predictions.metrics  # Evaluation metric  
    print("Metrics:", metrics)
    print(f"Completed experiment {peft_model_id}")
    print("----------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="1-8-0.1")
    args = parser.parse_args()

    main(args)
