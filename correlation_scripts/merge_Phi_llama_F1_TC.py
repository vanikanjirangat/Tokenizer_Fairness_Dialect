import pandas as pd

# Replace these with your actual file paths
phi_path = "metrics_output_llama3.csv"
llama_path = "metrics_output1.csv"

# Load the data
df_phi = pd.read_csv(phi_path)
df_llama = pd.read_csv(llama_path)

# Keep only 'lang' and 'macro_f1', rename macro_f1 to identify model
df_phi = df_phi[['lang', 'macro_f1']].rename(columns={'macro_f1': 'macro_f1_phi'})
df_llama = df_llama[['lang', 'macro_f1']].rename(columns={'macro_f1': 'macro_f1_llama'})

# Merge on 'lang'
merged_df = pd.merge(df_phi, df_llama, on='lang', how='outer')  # Use 'inner' if you only want common langs

# Optional: sort by language
merged_df = merged_df.sort_values(by='lang').reset_index(drop=True)

# Save or display
print(merged_df)
merged_df.to_csv("merged_macro_f1_TC_Phi_llama.csv", index=False)

