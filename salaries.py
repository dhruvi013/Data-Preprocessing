import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load the dataset
salaries_data = pd.read_csv("C:\\Users\\DREAMWORLD\\Downloads\\PreProcessing\\salaries.csv")

# Encode categorical columns
categorical_columns = [
    'experience_level', 'employment_type', 'job_title', 
    'salary_currency', 'employee_residence', 'company_location', 'company_size'
]
label_encoder = LabelEncoder()
for col in categorical_columns:
    salaries_data[col] = label_encoder.fit_transform(salaries_data[col])

# Scale numerical features
numerical_columns = ['work_year', 'salary', 'salary_in_usd', 'remote_ratio']
scaler = MinMaxScaler()
salaries_data[numerical_columns] = scaler.fit_transform(salaries_data[numerical_columns])

# Display preprocessed data
print("Preprocessed Dataset Head:")
print(salaries_data.head())

# Dataset statistics after preprocessing
print("\nPreprocessed Dataset Statistics:")
print(salaries_data.describe())

# Save the preprocessed dataset
salaries_data.to_csv("preprocessed_salaries.csv", index=False)
print("\nPreprocessed dataset saved as 'preprocessed_salaries.csv'")
