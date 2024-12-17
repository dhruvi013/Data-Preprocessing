import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load the dataset
data = pd.read_csv("C:\\Users\\DREAMWORLD\\Downloads\\PreProcessing\\survey lung cancer.csv")

# Encode categorical features
label_encoder = LabelEncoder()
data['GENDER'] = label_encoder.fit_transform(data['GENDER'])
data['LUNG_CANCER'] = label_encoder.fit_transform(data['LUNG_CANCER'])

# Scale numerical features
numerical_features = data.columns.drop(['LUNG_CANCER'])  # Exclude the target column
scaler = MinMaxScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Display preprocessed dataset
print("Preprocessed Dataset Head:")
print(data.head())

# Statistics after preprocessing
print("\nPreprocessed Dataset Statistics:")
print(data.describe())

# Save the preprocessed dataset to a new file
data.to_csv("preprocessed_survey_lung_cancer.csv", index=False)
print("\nPreprocessed dataset saved as 'preprocessed_survey_lung_cancer.csv'")
