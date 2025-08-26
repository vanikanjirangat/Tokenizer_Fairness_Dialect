from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
#languages = ['arabic-sau', 'english-kenya', 'swahili-kenya']  # example dialect folders
base_model = "bert-base-multilingual-cased"
import os
from sklearn.metrics import f1_score, accuracy_score
label_encoder = LabelEncoder()

lang_code_to_flores_key = {
    "en": "eng_Latn",# reference
    "ru": "rus_Cyrl",
    "fr": "fra_Latn",#French
    "ko": "kor_Hang",#Korean
    "ja": "jpn_Jpan",
    "he": "heb_Hebr",
    "hu": "hun_Latn",
    "no": "nob_Latn",
    "hi": "hin_Deva",# Hindi
    "fi": "fin_Latn",#Finnish
    "es": "spa_Latn",#Spanish
    "de": "deu_Latn",#German
    "it": "ita_Latn",
    "nl": "nld_Latn",
    "zh": "zho_Hans",
    "vi": "vie_Latn",
    "id": "ind_Latn",#Indonesian
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
    "pr": "por_Latn",
    "ht": "hat_Latn",
    "qu": "quy_Latn",
    "aj": "azj_Latn",
    "ck": "ckb_Arab",
    "nl": "nld_Latn",#Dutch
    "iv": "lvs_Latn",
    "lj": "lij_Latn",
    "zh": "zho_Hans",
    "ns": "nso_Latn",
}
lang_codes= list(lang_code_to_flores_key.values())
print(lang_codes)
lang_codes=['spa_Latn' 'ell_Grek' 'hat_Latn' 'srp_Cyrl' 'pan_Guru' 'eus_Latn'
 'fin_Latn' 'kan_Knda' 'tur_Latn' 'guj_Gujr' 'hin_Deva' 'bul_Cyrl'
 'mya_Mymr' 'swh_Latn' 'por_Latn' 'dan_Latn' 'hun_Latn' 'heb_Hebr'
 'quy_Latn' 'npi_Deva' 'tam_Taml' 'ben_Beng' 'zho_Hans' 'kat_Geor'
 'zsm_Latn' 'ces_Latn' 'nob_Latn' 'mal_Mlym' 'asm_Beng' 'urd_Arab'
 'ind_Latn' 'rus_Cyrl' 'ron_Latn' 'nld_Latn' 'vie_Latn' 'tel_Telu'
 'est_Latn' 'cat_Latn' 'ita_Latn' 'kor_Hang' 'fra_Latn' 'tha_Thai'
 'mar_Deva' 'hrv_Latn' 'deu_Latn' 'arb_Arab' 'ory_Orya' 'ukr_Cyrl'
 'jpn_Jpan']
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1_macro = f1_score(labels, preds, average='macro')
    f1_micro = f1_score(labels, preds, average='micro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro
    }

base_data_path = "./DialectBench/data/topic_class/"  # this should be the parent directory containing all language folders
languages = [name for name in os.listdir(base_data_path)
             if os.path.isdir(os.path.join(base_data_path, name))]
print(languages)
def preprocess(examples):
    return tokenizer(examples['sentence'], truncation=True, padding=True, max_length=128)
for lang in languages:
    #print(lang)
    if lang in lang_codes:
        print(lang)
        print(f"\nTraining for language: {lang}")
        data_dir = f"./DialectBench/data/topic_class/{lang}"  # Folder with train.csv, dev.csv, test.csv

        # Load CSVs
        train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
        val_df = pd.read_csv(os.path.join(data_dir, "dev.csv"))
        test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
        train_df['label'] = label_encoder.fit_transform(train_df['label'])
        val_df['label'] = label_encoder.transform(val_df['label'])
        test_df['label'] = label_encoder.transform(test_df['label'])
        label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        # Convert to HuggingFace Datasets
        train_ds = Dataset.from_pandas(train_df)
        val_ds = Dataset.from_pandas(val_df)
        test_ds = Dataset.from_pandas(test_df)

        # Tokenizer and model
        tokenizer = BertTokenizer.from_pretrained(base_model)
        model = BertForSequenceClassification.from_pretrained(base_model, num_labels=train_df['label'].nunique())

    #def preprocess(examples):
     #   return tokenizer(examples['sentence'], truncation=True, padding=True, max_length=128)

        train_ds = train_ds.map(preprocess, batched=True)
        val_ds = val_ds.map(preprocess, batched=True)
        test_ds = test_ds.map(preprocess, batched=True)

        # Training setup
        output_dir = f"./checkpoints/{lang}"
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            load_best_model_at_end=True,
            logging_dir=f"{output_dir}/logs",
            save_total_limit=2
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        if not os.path.exists(output_dir):
            print("YES")
            # Train and evaluate
            trainer.train()
            metrics = trainer.evaluate(test_ds)
            print(f"Test metrics for {lang}:", metrics)
            metrics['lang'] = lang
            # Save final model
            trainer.save_model(f"{output_dir}/final_model")
            metrics_df = pd.DataFrame([metrics])
            output_csv = "evaluation_mBERT_TC_nw.csv"
            if os.path.exists(output_csv):
                metrics_df.to_csv(output_csv, mode='a', index=False, header=False)
            else:
                metrics_df.to_csv(output_csv, index=False)
