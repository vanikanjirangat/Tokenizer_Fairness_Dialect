import pandas as pd
import datasets
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from datasets import Dataset
import json
ZERO_SHOT_CLASSIFIER_PROMPT = """ You are text classifier, who needs to classify the given Arabic sentence input into the specific dialect category. Given the sentence, analyze the dialectal features and try to classify the sentence into one of the 18 dialects listed, where the dialects are separated by commas: {classes}

Sentence: ```{sentence}```
Class:

From the above list of dialects, strictly select only one dialect that the provided Arabic sentence can be classified into. Do not predict anything else.
"""

ZERO_SHOT_CLASSIFIER_PROMPT1 = """Classify the sentence into one of 18 dialects. The list of dialects is provided below, where the classes are separated by commas: 

{classes}

From the above list of dialect, select only one dialect that the provided sentence can be classified into. The sentence will be delimited with triple backticks. Once again, only predict the dialect from the given list of dialects. Do not predict anything else.
Sentence: ```{sentence}```
Class:
"""
FEW_SHOT_CLASSIFIER_PROMPT = """Classify the sentence into one of 18 dialects. The list of classes is provided below, where the classes are separated by commas:

{classes}

From the above list of classes, select only one class that the provided sentence can be classified into. Once again, only predict the class from the given list of classes. Do not predict anything else. The sentence will be delimited with triple backticks. To help you, examples are provided of sentence and the corresponding class they belong to.

{few_shot_samples}

Sentence: ```{sentence}```
Class:
"""

TRAINING_CLASSIFIER_PROMPT = """Classify the following sentence that is delimited with triple backticks.

Sentence: ```{sentence}```
Class: {label}
"""

INFERENCE_CLASSIFIER_PROMPT = """Classify the following sentence that is delimited with triple backticks.

Sentence: ```{sentence}```
Class: 
"""

#TRAINING_CLASSIFIER_PROMPT_v2 = """What is the topic of the following Arabic text?\nSentence:{sentence}\nClass:{label}"""
#INFERENCE_CLASSIFIER_PROMPT_v2 = """[INST] Classify the topic of the following Arabic sentence:\n\n{sentence}\n\n[/INST] Class:"""


TRAINING_CLASSIFIER_PROMPT_v2 = """[INST]Extract the answer of the question from the given context\nQuestion:{sentence}\nContext:{context}\nAnswer:{label}[/INST]"""
INFERENCE_CLASSIFIER_PROMPT_v2 = """[INST] Answer the question based on given context. Output from the given context only as in extractive QA.
Question:
{sentence}
Context:{context}
[/INST]
Answer:"""

ZERO_SHOT_SUMMARIZATION_PROMPT = """Summarize the following dialogue that is delimited with triple backticks.

Dialogue: ```{dialogue}```
Summary:
"""

FEW_SHOT_SUMMARIZATION_PROMPT = """Summarize the following dialogue that is delimited with triple backticks. To help you, examples of summarization are provided.

{few_shot_samples}

Dialogue: ```{dialogue}```
Summary:
"""

TRAINING_SUMMARIZATION_PROMPT = """Summarize the following dialogue that is delimited with triple backticks.

Dialogue: ```{dialogue}```
Summary: {summary}
"""

TRAINING_SUMMARIZATION_PROMPT_v2 = """###Dialogue:{dialogue}###Summary:{summary}"""
INFERENCE_SUMMARIZATION_PROMPT_v2 = """###Dialogue:{dialogue}###Summary:"""

INFERENCE_SUMMARIZATION_PROMPT = """Summarize the following dialogue that is delimited with triple backticks.

Dialogue: ```{dialogue}```
Summary: 
"""


def get_instruction_data(mode, texts,contexts, labels):
    if mode == "train":
        prompt = TRAINING_CLASSIFIER_PROMPT_v2
    elif mode == "inference":
        prompt = INFERENCE_CLASSIFIER_PROMPT_v2

    instructions = []
    #allowed_labels=["science/technology", "travel", "politics", "sports", "health", "entertainment", "geography"]
    #label_str = ", ".join(allowed_labels)
    for text, context,label in zip(texts, contexts, labels):
        if mode == "train":
            example = prompt.format(
                sentence=text,
                context=context,
                label=label,
            )
        elif mode == "inference":
            #print("In inference")
            example = prompt.format(
                sentence=text,
                context=context,
                #allowed_labels=label_str
            )
        instructions.append(example)

    return instructions


def clean_data(texts, contexts,labels):
    label2data = {}
    clean_data, clean_contexts,clean_labels = [], [], []
    for data, context,label in zip(texts, contexts,labels):
        #if isinstance(data, str) and isinstance(label, str):
        clean_data.append(data)
        clean_labels.append(label)
        clean_contexts.append(context)
        if label not in label2data:
            label2data[label] = data

    return label2data, clean_data, clean_contexts,clean_labels


def get_newsgroup_data_for_ft(mode="train", train_sample_fraction=0.99):
    newsgroup_dataset = load_dataset("rungalileo/20_Newsgroups_Fixed")
    train_data = newsgroup_dataset["train"]["text"]
    train_labels = newsgroup_dataset["train"]["label"]
    label2data, train_data, train_labels = clean_newsgroup_data(
        train_data, train_labels
    )

    test_data = newsgroup_dataset["test"]["text"]
    test_labels = newsgroup_dataset["test"]["label"]
    _, test_data, test_labels = clean_newsgroup_data(test_data, test_labels)

    # sample n points from training data
    train_df = pd.DataFrame(data={"text": train_data, "label": train_labels})
    train_df, _ = train_test_split(
        train_df,
        train_size=train_sample_fraction,
        stratify=train_df["label"],
        random_state=42,
    )
    train_data = train_df["text"]
    train_labels = train_df["label"]

    train_instructions = get_newsgroup_instruction_data(mode, train_data, train_labels)
    test_instructions = get_newsgroup_instruction_data(mode, test_data, test_labels)

    train_dataset = datasets.Dataset.from_pandas(
        pd.DataFrame(
            data={
                "instructions": train_instructions,
                "labels": train_labels,
            }
        )
    )
    test_dataset = datasets.Dataset.from_pandas(
        pd.DataFrame(
            data={
                "instructions": test_instructions,
                "labels": test_labels,
            }
        )
    )

    return train_dataset, test_dataset


def get_newsgroup_data():
    newsgroup_dataset = load_dataset("rungalileo/20_Newsgroups_Fixed")
    train_data = newsgroup_dataset["train"]["text"]
    train_labels = newsgroup_dataset["train"]["label"]

    label2data, clean_data, clean_labels = clean_newsgroup_data(
        train_data, train_labels
    )
    df = pd.DataFrame(data={"text": clean_data, "label": clean_labels})

    newsgroup_classes = df["label"].unique()
    newsgroup_classes = ", ".join(newsgroup_classes)

    few_shot_samples = ""
    for label, data in label2data.items():
        sample = f"Sentence: {data} \n Class: {label} \n\n"
        few_shot_samples += sample

    return newsgroup_classes, few_shot_samples, df


def get_samsum_data():
    samsum_dataset = load_dataset("samsum")
    train_dataset = samsum_dataset["train"]
    dialogues = train_dataset["dialogue"][:2]
    summaries = train_dataset["summary"][:2]

    few_shot_samples = ""
    for dialogue, summary in zip(dialogues, summaries):
        sample = f"Sentence: {dialogue} \n Summary: {summary} \n\n"
        few_shot_samples += sample

    return few_shot_samples
    
    
def get_tc_data():
    #train_df = pd.read_csv("./data/gdi/train.txt",delimiter='\t', header=None, names=['sentence','label'])
    #train_df=train_df.drop(["#1_id"],axis=1)
    #train_df = train_df[~train_df.label.isin(['XY'])]
    #train_df=train_df.rename(columns={"sentence": "text", "label": "label"})
    #test_df = pd.read_csv("./data/gdi/gold.txt",delimiter='\t', header=None, names=['sentence','label'])
    #test_df=test_df.drop(["#1_id"],axis=1)
    #test_df=test_df.rename(columns={"sentence": "text", "label": "label"})
    #train_df = pd.read_csv("./data/topic_class/ace_Arab/train.csv",delimiter='\t', header=None, names=['index_id','label','sentence'])
    train_df = pd.read_csv("./data/topic_class/ace_Arab/train.csv", encoding='utf-8')
    print(train_df[:5])
    train_df=train_df.drop(["index_id"],axis=1)

    train_df=train_df.rename(columns={"sentence": "text", "label": "label"})

    dev_df = pd.read_csv("./data/topic_class/ace_Arab/dev.csv", encoding='utf-8')
   # dev_df = pd.read_csv("./data/topic_class/ace_Arab/dev.csv",delimiter='\t', header=None, names=['index_id','label','sentence'])
    dev_df=dev_df.drop(["index_id"],axis=1)
    dev_df=dev_df.rename(columns={"sentence": "text", "label": "label"})

    test_df = pd.read_csv("./data/topic_class/ace_Arab/test.csv", encoding='utf-8')
    #test_df = pd.read_csv("./data/topic_class/ace_Arab/test.csv",delimiter='\t', header=None, names=['index_id','label','sentence'])
    test_df=test_df.drop(["index_id"],axis=1)
    test_df=test_df.rename(columns={"sentence": "text", "label": "label"})
    print(train_df[:5])

    dataset={}
    dataset['train']=Dataset.from_pandas(train_df)
    dataset['test'] = Dataset.from_pandas(test_df)
    dataset=datasets.DatasetDict(dataset)
    #print(dataset)
    # ds_train = load_dataset("csv", data_files="/data/NADI2023_Subtask1_TRAIN.tsv",delimiter='\t')
    # di_dataset = load_dataset("rungalileo/20_Newsgroups_Fixed")
    train_data = dataset["train"]["text"]
    train_labels = dataset["train"]["label"]

    label2data, clean_tr, clean_labels = clean_data(
        train_data, train_labels
    )
    df = pd.DataFrame(data={"text": clean_tr, "label": clean_labels})

    classes = df["label"].unique()
    classes = ", ".join(classes)
    print(f'Classes:{classes}')
    few_shot_samples = ""
    for label, data in label2data.items():
        sample = f"Sentence: {data} \n Class: {label} \n\n"
        few_shot_samples += sample
    #print(few_shot_samples)
    return classes, few_shot_samples, df,dataset
def json_to_pd(data):
    records = []

    for entry in data['data']:
        for paragraph in entry['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                # Extract the start index and answer text
                answer = qa['answers'][0]['text']

                # Store the data in the list of records
                records.append({
                    'question': question,
                    'context': context,
                    'label': answer  # The 'label' corresponds to the answer text
                })
    r_df = pd.DataFrame(records)
    return r_df

def get_qa_data_for_ft(file_path,mode="train"):
    #qa_dir = "./DialectBench/data/Question-Answering/SDQA-gold-task"
    #qa_files = glob.glob(f"{qa_dir}/sdqa-dev-*.json")
    #features: ['id', 'title', 'context', 'question', 'answers']
    #print(folders)
    #for file_path in qa_files:
        #lang_code = os.path.basename(file_path).replace("sdqa-dev-", "").replace(".json", "")
        #print(f"\nFine-tuning for language: {lang_code}")
        #print(i)
        #dataset = load_dataset("json", data_files={"train": file_path}, field="data")
        #dataset = flatten_squad("Question-Answering/SDQA-gold-task/sdqa-dev-arabic-bhr.json")
    #raw_dataset = load_dataset("json", data_files={"train": file_path})
    #flat_dataset = flatten_squad_dataset(raw_dataset["train"])
    #print(flat_dataset)
    #print(flat_dataset.column_names)    
    #train_df = pd.read_csv("./data/topic_class/ace_Arab/train.csv",delimiter='\t', header=None, names=['index_id','label','sentence'])
    #train_df = pd.read_csv(f"./data/topic_class/{folder}/train.csv", encoding='utf-8')
    #print(train_df[:5])
    #train_df=train_df.drop(["index_id"],axis=1)
    with open(file_path, 'r') as f:
        data = json.load(f)
    train_df=json_to_pd(data)
    train_df=train_df.rename(columns={"question": "text", "answer": "label"})

    #dev_df = pd.read_csv(f"./data/topic_class/{folder}/dev.csv", encoding='utf-8')
   # dev_df = pd.read_csv("./data/topic_class/ace_Arab/dev.csv",delimiter='\t', header=None, names=['index_id','label','sentence'])
    #dev_df=dev_df.drop(["index_id"],axis=1)
    #dev_df=dev_df.rename(columns={"sentence": "text", "label": "label"})
    
    #test_df = pd.read_csv(f"./data/topic_class/{folder}/test.csv", encoding='utf-8')
    #test_df = pd.read_csv("./data/topic_class/ace_Arab/test.csv",delimiter='\t', header=None, names=['index_id','label','sentence'])
    #test_df=test_df.drop(["index_id"],axis=1)
    #test_df=test_df.rename(columns={"sentence": "text", "label": "label"})
    print(train_df[:1])
    
    dataset={}
    dataset['train']=Dataset.from_pandas(train_df)
    #dataset['dev']=Dataset.from_pandas(dev_df)
    #dataset['test'] = Dataset.from_pandas(test_df)
    dataset=datasets.DatasetDict(dataset)
    print(dataset)
    train_data = dataset["train"]["text"]
    train_labels = dataset["train"]["label"]
    train_context = dataset["train"]["context"]
    #dev_data = dataset["dev"]["text"]
    #dev_labels = dataset["dev"]["label"]
    #test_data = dataset["test"]["text"]
    #test_labels = dataset["test"]["label"]

    print(f'train_data:{train_data[:1]}')
    print(f'train_labels:{train_labels[:1]}')
    print(f'train_context:{train_context[:1]}')
    label2data, train_data, train_context,train_labels = clean_data(
        train_data,train_context, train_labels
    )
    #label2data, dev_data, dev_labels = clean_data(
     #   dev_data, dev_labels
    #)
    #_, test_data, test_labels = clean_data(test_data, test_labels)



    print(f'train_data:{train_data[:1]}')
    print(f'train_labels:{train_labels[:1]}')
    print(f'train_context:{train_context[:1]}')
    train_instructions = get_instruction_data(mode, train_data, train_context,train_labels)
    #dev_instructions = get_instruction_data(mode, dev_data, dev_labels)
    #test_instructions = get_instruction_data(mode, test_data, test_labels)
    
    print(f'train_data:{train_instructions[:1]}')
    #print(f'train_labels:{train_labels[:5]}')

    train_dataset = datasets.Dataset.from_pandas(
        pd.DataFrame(
            data={
                "instructions": train_instructions,
                "labels": train_labels,
            }
        )
    )
    #dev_dataset = datasets.Dataset.from_pandas(
     #   pd.DataFrame(
      #      data={
       #         "instructions": dev_instructions,
        #        "labels": dev_labels,
         #   }
        #)
    #)
    #test_dataset = datasets.Dataset.from_pandas(
     
     #   pd.DataFrame(
      #      data={
       #         "instructions": test_instructions,
        #        "labels": test_labels,
         #   }
        #)
    #)
    
    ds={}
    ds["train"]=train_dataset
    #ds["dev"]=dev_dataset
    #ds["test"]=test_dataset
    
    return train_dataset,ds
