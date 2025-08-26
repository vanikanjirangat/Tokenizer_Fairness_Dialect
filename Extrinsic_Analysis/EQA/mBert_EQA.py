from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
from datasets import load_dataset, DatasetDict, Dataset
from pathlib import Path
import os
import json
import pandas as pd

from sklearn.metrics import f1_score, accuracy_score
from transformers import default_data_collator

model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

#from datasets import load_metric
from evaluate import load
import numpy as np

squad_metric = load("squad")

data_dir = Path("DialectBench/data/Question-Answering/SDQA-gold-task/")  # Your dataset folder
json_files = sorted(data_dir.glob("sdqa-dev-*.json"))  # dev files
results = []
def postprocess_qa_predictions(examples, features, raw_predictions):
    predictions = []

    all_start_logits, all_end_logits = raw_predictions
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = {example_id_to_index[feat["id"]]: [] for feat in features}

    for i, feat in enumerate(features):
        features_per_example[example_id_to_index[feat["id"]]].append(i)

    for example_index, example in enumerate(examples):
        feature_indices = features_per_example[example_index]

        context = example["context"]
        answers = []

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]

            offset_mapping = features[feature_index]["offset_mapping"]
            input_ids = features[feature_index]["input_ids"]

            start_indexes = np.argsort(start_logits)[-1:-21:-1].tolist()
            end_indexes = np.argsort(end_logits)[-1:-21:-1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    if end_index < start_index or end_index - start_index + 1 > 30:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    predicted_answer = context[start_char:end_char]
                    answers.append(predicted_answer)

        if len(answers) > 0:
            predictions.append(answers[0])
        else:
            predictions.append("")

    return {k: v for k, v in zip(examples["id"], predictions)}

for dev_file in json_files:
    lang_code = dev_file.stem.replace("sdqa-dev-", "")
    test_file = dev_file.with_name(dev_file.name.replace("dev", "test"))

    # Load data in SQuAD format
    dataset = load_dataset("json", data_files={"train": str(dev_file), "test": str(test_file)}, field="data")

    # Preprocess function for QA
    def prepare_qa(examples):
        tokenized = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=384,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length"
        )

        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized.pop("offset_mapping")

        start_positions = []
        end_positions = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]

            if len(answers["answer_start"]) == 0:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                sequence_ids = tokenized.sequence_ids(i)
                context_index = 1

                token_start_index = 0
                while sequence_ids[token_start_index] != context_index:
                    token_start_index += 1

                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != context_index:
                    token_end_index -= 1

                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    start_positions.append(cls_index)
                    end_positions.append(cls_index)
                else:
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    start_positions.append(token_start_index - 1)

                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    end_positions.append(token_end_index + 1)

        tokenized["start_positions"] = start_positions
        tokenized["end_positions"] = end_positions
        return tokenized

    tokenized_ds = dataset.map(prepare_qa, batched=True, remove_columns=dataset["train"].column_names)

    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    training_args = TrainingArguments(
        output_dir=f"./checkpoints/mbert_qa_{lang_code}",
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        logging_dir=f"./logs_{lang_code}",
        save_total_limit=1,
        save_strategy="epoch",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        tokenizer=tokenizer,
        data_collator=default_data_collator
    )

    trainer.train()

    # Evaluate
    #eval_results = trainer.evaluate()
    #eval_results["lang"] = lang_code
    #results.append(eval_results)
    raw_predictions = trainer.predict(tokenized_ds["test"])
    predictions = postprocess_qa_predictions(
        dataset["test"], tokenized_ds["test"], raw_predictions.predictions
    )

    # Format references and predictions
    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
    references = [
        {"id": ex["id"], "answers": ex["answers"]} for ex in dataset["test"]
    ]

    metrics = squad_metric.compute(predictions=formatted_predictions, references=references)
    metrics["lang"] = lang_code
    results.append(metrics)
    # Save model
    trainer.save_model(f"./models/mbert_qa_{lang_code}")

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("mbert_qa_lang_eval_results.csv", index=False)

