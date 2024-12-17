import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
file_path = 'C:\\Users\\DREAMWORLD\\Downloads\\PreProcessing\\spotify_tracks (1).csv'
dataset = pd.read_csv(file_path)

# Step 1: Inspect the dataset
print("Initial Dataset Info:")
print(dataset.info())
print("\nFirst 5 rows of the dataset:")
print(dataset.head())

# Step 2: Check for missing values
print("\nMissing Values:")
print(dataset.isnull().sum())

# Step 3: Remove duplicates
duplicate_count = dataset.duplicated().sum()
print(f"\nDuplicate Rows: {duplicate_count}")
dataset_cleaned = dataset.drop_duplicates()

# Step 4: Normalize numerical columns
numerical_columns = dataset_cleaned.select_dtypes(include=['float64', 'int64']).columns
scaler = MinMaxScaler()
dataset_cleaned[numerical_columns] = scaler.fit_transform(dataset_cleaned[numerical_columns])

# Step 5: Encode categorical columns
dataset_encoded = pd.get_dummies(dataset_cleaned, columns=['language'], drop_first=True)

# Step 6: Drop irrelevant columns
columns_to_drop = ['artwork_url', 'track_url']
dataset_final = dataset_encoded.drop(columns=columns_to_drop)

# Save results
original_file_path = 'C:\\Users\\DREAMWORLD\\Downloads\\PreProcessing\\spotify_original.csv'
processed_file_path = 'C:\\Users\\DREAMWORLD\\Downloads\\PreProcessing\\spotify_processed.csv'
dataset.to_csv(original_file_path, index=False)
dataset_final.to_csv(processed_file_path, index=False)

print("\nData Preprocessing Completed.")
print(f"Original data saved to: {original_file_path}")
print(f"Processed data saved to: {processed_file_path}")
