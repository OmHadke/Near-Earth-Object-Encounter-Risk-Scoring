import pandas as pd

# orbital_df = pd.read_csv("data/orbital_elements.csv")
# print(orbital_df.head())
# print(orbital_df.columns)

# # Load the first dataset from the 'data' folder
cneos = pd.read_csv("data/orbital_elements.csv") 

# # Load the second dataset from the 'data' folder
nea = pd.read_csv("data/nea_feed_large.csv")
# # Normalize names
cneos['full_name'] = cneos['full_name'].str.replace(r'[\(\)]', '', regex=True).str.strip().str.lower()
nea['name'] = nea['name'].str.replace(r'[\(\)]', '', regex=True).str.strip().str.lower()

# # Merge
merged = pd.merge(cneos, nea, left_on='full_name', right_on='name', how='inner')

# # Save the merged DataFrame inside the 'data' folder
merged.to_csv('data/merged_neo_data.csv', index=False)
merged = pd.read_csv('data/merged_neo_data.csv')
print("Merged dataset shape:", merged.shape)
print(merged.columns)
print(merged.head())
