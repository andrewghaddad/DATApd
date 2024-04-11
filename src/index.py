import pandas as pd

# Load the dataset
data = pd.read_csv("./src/data.csv")

# Display the first few rows of the dataset
print("Original Dataset:")
print(data.head())

# Data Cleaning
# Remove duplicate rows
data = data.drop_duplicates()

# Handle missing values
# Replace missing values in numeric columns with mean
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# Replace missing values in categorical columns with mode
categorical_cols = data.select_dtypes(include=['object']).columns
data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])

# Data Preprocessing
# # Convert categorical variables to numerical using one-hot encoding
data = pd.get_dummies(data, columns=categorical_cols)

# Normalize numerical features
normalized_data = (data - data.min()) / (data.max() - data.min())

# Save the cleaned and preprocessed dataset
normalized_data.to_csv("cleaned_healthcare_data.csv", index=False)

# Display the cleaned and preprocessed dataset
print("\nCleaned and Preprocessed Dataset:")
print(normalized_data.head())
