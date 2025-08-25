from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import f1_score
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import classification_report
import torch
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
import pandas as pd
import datasets
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification
classes=['ES-AR','ES-ES']
data=pd.read_csv('./train_ES.csv')
print(data.head())
os.environ["WANDB_PROJECT"] = "Dialect Identifications"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints
def get_di_data_ML_FT():
    #data=pd.read_csv('./train_FR.csv')
    data=pd.read_csv('./train_ES.csv')
    sentences=data["Text"].values
    l=data.iloc[:,2:]
    labs=list(l.columns)
    #map={'n':0,'y':1}
    #data[labs]=data[labs].replace(map)
    # Create an empty list 
    label_list =[]
    # Iterate over each row 
    #FR-BE  FR-CA  FR-CH  FR-FR
    for index, rows in data.iterrows():
        # Create list for the current row 
        #my_list =[rows.FR-BE,rows.FR-CA,rows.FR-CH,rows.FR-FR]
        my_list =[rows["ES-AR"],rows["ES-ES"]]
        # append the list to the final list 
        label_list.append(my_list)
    #c_names=[["sentence"]+COUNTRIES]
    #df = pd.DataFrame(Row_list, columns = c_names) 
    dict={"text":sentences,"label":label_list}
    train_df=pd.DataFrame(dict)
    print("train_df",len(train_df))
	
    data=pd.read_csv('./dev_ES.csv')
    #data=pd.read_csv('./dev_filt_multi_ES.csv')
    sentences=data["Text"].values
    l=data.iloc[:,2:]
    labs=list(l.columns)
    #map={'n':0,'y':1}
    #data[labs]=data[labs].replace(map)
    # Create an empty list 
    label_list =[]
    # Iterate over each row 
    #FR-BE  FR-CA  FR-CH  FR-FR
    for index, rows in data.iterrows():
        # Create list for the current row 
        #my_list =[rows.FR-BE,rows.FR-CA,rows.FR-CH,rows.FR-FR]
        my_list =[rows["ES-AR"],rows["ES-ES"]]
        # append the list to the final list 
        label_list.append(my_list)
    #c_names=[["sentence"]+COUNTRIES]
    #df = pd.DataFrame(Row_list, columns = c_names) 
    dict={"text":sentences,"label":label_list}
    test_df=pd.DataFrame(dict)
    print("test_df",len(test_df))
    dataset={}
    dataset['train']=Dataset.from_pandas(train_df)
    dataset['test'] = Dataset.from_pandas(test_df)
    dataset=datasets.DatasetDict(dataset)
    print(dataset)
    return dataset


# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)
# Define a compute_metrics function for multilabel evaluation
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def compute_metrics(p):
    logits, labels = p
    sigmoid_logits = torch.sigmoid(torch.tensor(logits))
    preds = (sigmoid_logits > 0.5).float()  # Multilabel threshold for predictions
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
        "precision": precision_score(labels, preds, average="weighted"),
        "recall": recall_score(labels, preds, average="weighted")
    }

ds=get_di_data_ML_FT()
train_dataset=ds["train"]
eval_dataset=ds["test"]

#print(train_dataset[:2])

dataset=ds
#model_path="chriskhanhtran/spanberta"
model_path="bert-base-uncased"
#model_path="google-bert/bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=True)
print("ds",ds)
# Apply tokenization to the entire dataset
tokenized_ds = dataset.map(tokenize_function, batched=True)
print(tokenized_ds)
#tokenized_eval_ds = eval_dataset.map(tokenize_function, batched=True)
#tokenized_ds=tokenized_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
#tokenized_eval_ds=tokenized_eval_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
#print("TT",tokenized_ds)
# Load pre-trained BERT model and tokenizer
#model_name = "bert-base-uncased"
#tokenizer = BertTokenizer.from_pretrained(model_name)

#num_labels = 4
from transformers import Trainer, TrainingArguments

# Define custom loss function in the model
class CustomBERTForMultilabel(BertForSequenceClassification):
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        logits = outputs.logits

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())  # Convert labels to float for BCEWithLogits
            return {"loss": loss, "logits": logits}

        return outputs
#model = CustomBERTForMultilabel.from_pretrained(model_name, num_labels=num_labels)

model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    problem_type="multi_label_classification",
    num_labels=2
)


# Load the fine-tuned multilabel model
model = AutoModelForSequenceClassification.from_pretrained("./results_ESbert_base/checkpoint-1302",problem_type="multi_label_classification", num_labels=2)  #
#model = model.to('cpu')
#model=model.to("cuda") 
print("Loaded")
# Define training arguments
training_args = TrainingArguments(
    output_dir="./results_ESbert_base",
    #evaluation_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    #per_device_eval_batch_size=8,
    logging_dir='./logs',
    logging_steps=10,fp16=True,report_to="wandb",
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds['train'],
    #eval_dataset=tokenized_ds['test'],
    #compute_metrics=compute_metrics,
)
# Train the model
#trainer.train()

torch.cuda.empty_cache()
#trainer.evaluate()
tk=tokenized_ds['test']
print("$$",len(tk))
print("Predicting..")
print("TK",tk)
# Convert to PyTorch tensors
#input_ids = torch.tensor(tk['input_ids'])
#attention_mask = torch.tensor(tk['attention_mask'])
#labels=torch.tensor(tk['label'])

# Prepare the dataset in the expected format
#tk = torch.utils.data.TensorDataset(input_ids, attention_mask)
#print(tk)
with torch.no_grad():
    predictions = trainer.predict(tk)
logits = predictions.predictions
#print(logits)
import torch.nn.functional as F
probs = F.sigmoid(torch.tensor(logits))
#print(probs)
# Set threshold (e.g., 0.5) to decide if a label is active (1) or not (0)
threshold = 0.5
predicted_labels = (probs > threshold).int()  # Convert probabilities to 0s and 1s based on threshold
print("$$",len(predicted_labels))
#print("Predicted Labels (multilabel):", predicted_labels)
#true_labels=predictions.label
true_labels=torch.tensor(tk['label'])
print("$$",len(true_labels))
report = classification_report(true_labels, predicted_labels,target_names=['ES-AR','ES-ES'])
print(report)
pred_e=[str(x) for x in predicted_labels]
from collections import Counter
print(Counter(pred_e))
#report = multilabel_confusion_matrix(true_labels, predicted_labels, labels=['FR-BE','FR-CA','FR-CH','FR-FR'])
#print(report)
print(f1_score(y_true=true_labels, y_pred=predicted_labels, average='weighted'))

