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
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report
from datasets import Dataset, DatasetDict
import torch.nn as nn
import functools
import torch.nn.functional as F
import numpy as np
from peft import (
    PeftConfig,
    PeftModel,
)
from transformers import (
    AutoModelForCausalLM,AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,Trainer,TrainingArguments,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from prompts_ES  import get_di_data_ML_FT

metric = evaluate.load("rouge")
warnings.filterwarnings("ignore")
def compute_metrics(p):
    predictions, labels = p

    # Debugging output
    print("Raw Predictions:\n", predictions)
    print("Raw Labels:\n", labels)

    # Apply sigmoid to the predictions to get probabilities (if not done already)
    #probabilities = 1 / (1 + np.exp(-predictions))  # Sigmoid activation

    # Convert probabilities to binary predictions using a threshold
    threshold = 0.5
    binary_predictions = (probabilities > threshold).astype(int)

    # Debugging output after thresholding
    print("Binary Predictions:\n", binary_predictions)

    # Calculate F1 scores
    f1_micro = f1_score(labels, binary_predictions, average='micro')
    f1_macro = f1_score(labels, binary_predictions, average='macro')
    f1_weighted = f1_score(labels, binary_predictions, average='weighted')

    return {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }
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
class CustomTrainer1(Trainer):

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
        self.classification_head = nn.Linear(vocab_size, 2).to("cuda")# 4-- FR, 2--ES
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

def main(args):
    _, test_dataset,ds = get_di_data_ML_FT(mode="inference")
    #test_dataset=test_dataset[:2000]
    ds=DatasetDict(ds)
    experiment = args.experiment
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_ds = ds.map(functools.partial(tokenize_examples, tokenizer=tokenizer), batched=True)
    tokenized_ds = tokenized_ds.with_format('torch')
    tokenized_ds = tokenized_ds.map(
    lambda x: {'input_ids': x['input_ids'], 'attention_mask': x['attention_mask'], 'labels': x['labels']},remove_columns=tokenized_ds['test'].column_names)

    peft_model_id = f"{experiment}/checkpoint-2165"
    #checkpoint-2165"
    #./LLM_experiments/meta-llama/Llama-3.2-3B_classification_epochs-10_rank-8_dropout-0.1_ES/checkpoint-4330
    #peft_model_id = f"{experiment}/assets"
    config = PeftConfig.from_pretrained(peft_model_id)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model_name_or_path,
        quantization_config=bnb_config,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, peft_model_id)
    model.config.pad_token_id = model.config.eos_token_id
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    #tokenized_ds = ds.map(functools.partial(tokenize_examples, tokenizer=tokenizer))
    #tokenized_ds = tokenized_ds.with_format('torch')
    # Define training args
    training_args = TrainingArguments(
        logging_steps=100,
        #report_to="none",
        per_device_train_batch_size=1,
       # per_device_eval_batch_size =1,
        gradient_accumulation_steps=8,
        output_dir="./",learning_rate=5e-5,#changed
        num_train_epochs=10,logging_dir="./logs",
        fp16=True,
        #evaluation_strategy = 'epoch',
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",#changed
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        #save_strategy = 'epoch',
        #load_best_model_at_end = True,
        #report_to="wandb",
    )

    print(f"training_args = {training_args}")
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        #peft_config=peft_config,
        tokenizer=tokenizer,
        train_dataset=tokenized_ds['train'],
        #eval_dataset = tokenized_ds['test'],
        data_collator = functools.partial(collate_fn, tokenizer=tokenizer),
        #max_seq_length=512,
        #compute_metrics = compute_metrics,
        #dataset_text_field="instructions",
        #packing=True,
    )
    #tokenized_ds = tokenized_ds.map(
    #lambda x: {'input_ids': x['input_ids'], 'attention_mask': x['attention_mask'], 'labels': x['labels']},remove_columns=tokenized_ds['test'].column_names)  # Remove all original columns)
    print(tokenized_ds["test"])
    print(tokenized_ds["test"][0])
    ctr = 0
    results = []
    instructions, labels = test_dataset["instructions"], test_dataset["labels"]
    true=[]
    #print(trainer.evaluate(tokenized_ds['test']))
    #tokenized_ds = ds.map(functools.partial(tokenize_examples, tokenizer=tokenizer), batched=True)
    #tokenized_ds = tokenized_ds.with_format('torch')
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
            result=result.split(",")
            result=[x.strip() for x in result]
            #result=[int(x) for x in result if len(x)==1]
            f=all(isinstance(x,int) for x in result)
            print((result,label))
            if len(result)==4:
                #result=[x for x in result in type(x)=int]
                #f=all(isinstance(x,int) for x in result)
                #if f:
		#print((result,label))
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
    #['ES_AR','ES-ES']
    '''
    label_names =['ES_AR','ES-ES']
    #label_names = ['FR-BE','FR-CA','FR-CH','FR-FR']
    #t=tokenized_ds['test'].select(range(5))
    t=tokenized_ds["test"]
    preds=trainer.predict(t)
    # Get the predicted logits and true labels
    logits = preds.predictions  # Model outputs
    labels = preds.label_ids      # True labels
    print(logits)
    #logits = logits[:, -1, :]
    #print(logits)
    #print("$$$")
    #print(logits.shape[1])
    #vocab_size=logits.shape[1]
    # Add a classification head to map logits to num_labels
    #classification_head = nn.Linear(vocab_size, 2).to("cuda")# 4-- FR, 2--ES
    #logits = classification_head(logits)
    # Apply sigmoid to logits (multilabel classification)
    sigmoid_logits = torch.sigmoid(torch.tensor(logits))
    #print(logits)
    print(sigmoid_logits)
    # Convert sigmoid outputs to binary (multilabel predictions)
    predicted_labels = (sigmoid_logits > 0.5).int()
    # Convert logits to predicted labels (for classification tasks)
    #predicted_labels = np.argmax(logits, axis=1)
    print("***********")
    print((labels[:5],predicted_labels[:5]))
   # Generate a classification report
    report = classification_report(labels, predicted_labels, target_names=label_names)#print((labels[:5],predicted_labels[:5]))
    # Print the classification report
    print(report)
    pred_e=[str(x) for x in predicted_labels]
    print(list(set(pred_e)))
    from collections import Counter
    print (Counter(pred_e))
    #print(classification_report(labels,results,target_names=label_names))#print(confusion_matrix(labels,results))
   # predictions=results
    #from sklearn.metrics import accuracy_score
    '''
    for i,p in enumerate(predictions):
        if (p.count(1)) > 1:
            print("predicted multilabeled")
            print(p)
            print(labels[i])
    print(accuracy_score(true, predictions))
    #from sklearn.metrics import confusion_matrix
    #from sklearn.metrics import multilabel_confusion_matrix
    print(multilabel_confusion_matrix(true, predictions))
    #from sklearn.metrics import classification_report

    #label_names = ['FR-BE','FR-CA','FR-CH','FR-FR']
    print(len(true),len(predictions))
    print(classification_report(true, predictions,target_names=label_names))
    save_dir = os.path.join(f"{args.experiment}", "metrics")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, "metrics.pkl"), "wb") as handle:
        pickle.dump(metrics, handle)
    '''
    print(f"Completed experiment {peft_model_id}")
    print("----------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="1-8-0.1")
    args = parser.parse_args()

    main(args)
