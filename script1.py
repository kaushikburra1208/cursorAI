import os
# Get the current working directory
current_directory = os.getcwd()

# Construct the file path using the current directory
csv_file_path = os.path.join(current_directory, 'stock_info_data.csv')

# Check if the file exists
if not os.path.exists(csv_file_path):
    print(f"Error: The file '{csv_file_path}' does not exist.")
    print("Please make sure the CSV file is in the same directory as this script.")
    exit()

# If the file exists, update the csv_file_path variable
print(f"Using CSV file: {csv_file_path}")

# Ensure pandas is imported
import pandas as pd

# If the error persists, you can try reloading the module
import importlib
importlib.reload(pd)

df = pd.read_csv(csv_file_path)

print(df.head())

print(df.describe())

print(df.info())

# Perform exploratory data analysis

# Display basic information about the dataset
print("\nDataset Information:")
print(df.info())

# Display summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Display unique values in categorical columns (if any)
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    print(f"\nUnique values in {col}:")
    print(df[col].value_counts())

# Analyze numerical columns
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

# Calculate correlation matrix for numerical columns
correlation_matrix = df[numerical_columns].corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Identify important features based on correlation with target variable (assuming 'target' is the target column)
# Replace 'target' with your actual target column name
if 'target' in df.columns:
    target_correlations = correlation_matrix['target'].abs().sort_values(ascending=False)
    print("\nFeature Importance based on correlation with target:")
    print(target_correlations)

# Visualize distribution of numerical features
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 10))
df[numerical_columns].hist(bins=20, figsize=(15, 10))
plt.tight_layout()
plt.savefig('numerical_features_distribution.png')
plt.close()
print("\nHistograms of numerical features saved as 'numerical_features_distribution.png'")

# Visualize boxplots for numerical features
plt.figure(figsize=(15, 10))
df[numerical_columns].boxplot(figsize=(15, 10))
plt.title('Boxplots of Numerical Features')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('numerical_features_boxplot.png')
plt.close()
print("\nBoxplots of numerical features saved as 'numerical_features_boxplot.png'")

print("\nExploratory Data Analysis completed. Check the generated visualizations for more insights.")

# Feature Engineering

print("\nPerforming Feature Engineering:")

# One-hot encoding for categorical features
print("Applying one-hot encoding to categorical features...")
df_encoded = pd.get_dummies(df, columns=categorical_columns)

# Standardization for numerical features
from sklearn.preprocessing import StandardScaler

print("Applying standardization to numerical features...")
scaler = StandardScaler()
df_encoded[numerical_columns] = scaler.fit_transform(df_encoded[numerical_columns])

# Min-Max scaling for selected numerical features
from sklearn.preprocessing import MinMaxScaler

print("Applying Min-Max scaling to selected numerical features...")
# Select a subset of numerical features for Min-Max scaling
minmax_columns = numerical_columns[:len(numerical_columns)//2]  # Using first half of numerical columns as an example
minmax_scaler = MinMaxScaler()
df_encoded[minmax_columns] = minmax_scaler.fit_transform(df_encoded[minmax_columns])

# Frequency encoding for categorical features
print("Applying frequency encoding to categorical features...")
for col in categorical_columns:
    frequency = df[col].value_counts(normalize=True)
    df_encoded[f'{col}_freq'] = df[col].map(frequency)

# Binning for numerical features
print("Applying binning to selected numerical features...")
# Select a numerical feature for binning (e.g., the first numerical column)
bin_column = numerical_columns[0]
df_encoded[f'{bin_column}_binned'] = pd.qcut(df[bin_column], q=4, labels=['low', 'medium-low', 'medium-high', 'high'], duplicates="drop")

# Interaction features
print("Creating interaction features...")
# Create interaction features for the first two numerical columns
interaction_cols = numerical_columns[:2]
df_encoded['interaction'] = df_encoded[interaction_cols[0]] * df_encoded[interaction_cols[1]]

# Polynomial features
from sklearn.preprocessing import PolynomialFeatures

print("Creating polynomial features...")
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df_encoded[numerical_columns])
poly_columns = [f'poly_{i}' for i in range(poly_features.shape[1])]
df_encoded[poly_columns] = poly_features

print("\nFeature Engineering completed. New dataframe shape:", df_encoded.shape)
print("New features added:")
print(set(df_encoded.columns) - set(df.columns))

# Update numerical and categorical columns
numerical_columns = df_encoded.select_dtypes(include=['int64', 'float64']).columns
categorical_columns = df_encoded.select_dtypes(include=['object']).columns

print("\nUpdated number of numerical features:", len(numerical_columns))
print("Updated number of categorical features:", len(categorical_columns))

