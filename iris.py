import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load the dataset
iris_data = pd.read_csv("C:\\Users\\DREAMWORLD\\Downloads\\PreProcessing\\iris.csv")

# Encode the target variable
label_encoder = LabelEncoder()
iris_data['species'] = label_encoder.fit_transform(iris_data['species'])

# Scale the numerical features
numerical_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
scaler = MinMaxScaler()
iris_data[numerical_features] = scaler.fit_transform(iris_data[numerical_features])

# Display preprocessed data
print("Preprocessed Dataset Head:")
print(iris_data.head())

# Dataset statistics after preprocessing
print("\nPreprocessed Dataset Statistics:")
print(iris_data.describe())

# Save the preprocessed dataset
iris_data.to_csv("preprocessed_iris.csv", index=False)
print("\nPreprocessed dataset saved as 'preprocessed_iris.csv'")
