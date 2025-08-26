from io import StringIO
import pandas as pd
with open("./assets/tokenization_lengths_validated_nw.csv", 'r') as file:
    data = StringIO(file.read().replace('–––', ''))
df = pd.read_csv(data)
# df=df.replace('---', np.nan)
#df=df.set_index('Language')
#df=df.sort_index()
target_language = {
    "BERTbase": "English",
    "mBERT": "English",
    "MARBERT": "Standard Arabic",
    #"MARBERT": "English",
    "IndicBERT": "Hindi",
    #"IndicBERT":"English",
    "GermanBERT": "German",
    #"GermanBERT": "English",
    "SpaBERTa": "Spanish",
    #"SpaBERTa": "English",
    "CamemBERT": "French",
    #"CamemBERT":"English",
    "GreekBERT": "Greek",
    #"GreekBERT": "English",
    "Mixtral": "English",
    "Mistral": "English",
    "Falcon": "English",
    "Phi_Mini": "English",
    "Phi_MOE": "English",
    "Gemma": "English",
    "LLAMA": "English",
    "Llama3": "English",
    "SILMA": "Standard Arabic",
    #"SILMA": "English",
    "Meltemi": "Greek",
    #"Meltemi":"English",
    "BloomZ": "English",
    "Bloom": "English",
    "NLLB": "English",
    "mT5": "English",
    "FlanT5": "English",
    "ByT5":"English",
    "CANINE":"English",
    "BLOOM": "English",
    "ArabicBERT":"Standard Arabic",
    #"ArabicBERT":"English"
}
df = df.reset_index()
'''
for col in df.columns:
    df[col] /= df.loc[target_language[col], col]

for name in target_language.keys():
  print(name)
  print(df[name].reindex(df[name].index).reset_index())
  print("###########\n\n")
'''
print(df.columns)
import pandas as pd
import json
from io import StringIO

# Set Language as index
df_indexed = df.set_index("Language")
# Normalize each column by its respective target language row
for col in df.columns:
    if col in target_language:
        ref_lang = target_language[col]
        df[col] = df[col] / df.loc[df['Language'] == ref_lang, col].values[0]

# Re-index with language
df_indexed = df.set_index("Language")

# Convert to nested dict
tp_dict = {
    model: df_indexed[model].dropna().to_dict()
    for model in df_indexed.columns
}
#samples_per_language = df['Language'].value_counts()
#print(samples_per_language)

# Save to JSON (optional)
with open("token_parity_scores_target.json", "w") as f:
    json.dump(tp_dict, f, indent=2)
