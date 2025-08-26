import pandas as pd

# Define your resource classification (example lists, adjust based on your task)
high_resource_langs = {
    "eng", "spa", "fra", "deu", "ita", "por", "nld", "ron", "tur", "ces", "hun", "fin",
    "est", "hrv", "zsm", "cat", "vie", "rus", "jpn", "kor", "zho", "hin", "heb", "tha", "bul"
}
#Latin_high_resource={spa_Latn, nld_Latn, ita_Latn, nob_Latn, por_Latn, hrv_Latn,
#ron_Latn, est_Latn, fin_Latn, ces_Latn, hun_Latn, dan_Latn,
#cat_Latn, deu_Latn, fra_Latn, eus_Latn, tur_Latn, zsm_Latn}
#Latin_low_resource=
# Read the CSV file
#df = pd.read_csv("metrics_output1.csv")
df=pd.read_csv("metrics_output_llama3.csv")
print(df["lang"].values)
# Extract language family and script
df[['lang_id', 'script']] = df['lang'].str.split('_', expand=True)

# Define function to classify each language
def classify(row):
    script = row['script']
    lang_id = row['lang_id']
    latin = script == 'Latn'
    resource = 'high' if lang_id in high_resource_langs else 'low'
    return f"{'Latin' if latin else 'Non-Latin'}-{resource.capitalize()}"

# Apply classification
df['category'] = df.apply(classify, axis=1)

# Group by category and compute averages
category_avgs = df.groupby('category')[['micro_f1', 'macro_f1', 'precision', 'recall', 'accuracy']].mean()

# Display the result
print(category_avgs)
