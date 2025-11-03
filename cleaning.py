import pandas as pd

merged = pd.read_csv("data/cleaned_neo_data.csv")

# numeric_cols = [
#     "H", "albedo", "diameter_sigma", "e", "a", "i",
#     "absolute_magnitude_h", "estimated_diameter_min_km",
#     "estimated_diameter_max_km", "relative_velocity_km_per_s", "miss_distance_km"
# ]

# for col in numeric_cols:
#     merged[col] = pd.to_numeric(merged[col], errors="coerce")

# # Drop rows ONLY if they have more than 50% missing values
# merged = merged.dropna(thresh=len(merged.columns) * 0.5)

# # Fill the remaining missing numeric values with column mean
# merged = merged.fillna(merged.mean(numeric_only=True))

# merged.to_csv("data/cleaned_neo_data.csv", index=False)
# print("Cleaned dataset saved successfully!")
print("Cleaned dataset shape:", merged.shape)
print(merged.describe())
print(merged['is_potentially_hazardous'].value_counts())
