import pandas as pd
from sklearn.model_selection import train_test_split
import os
import numpy as np

# Define file paths
input_file = "sexism_classification_dataset.csv"  # Change this to your input file path
output_dir = "."            # Change this to your desired output directory
train_file = os.path.join(output_dir, "train.csv")
test_file = os.path.join(output_dir, "test.csv")

# Load the dataset
df = pd.read_csv(input_file)

# Check for NaN values in the label_vector column
nan_count = df['label_vector'].isna().sum()
if nan_count > 0:
    print(f"Warning: Found {nan_count} NaN values in 'label_vector' column")
    
    # Option 1: Remove rows with NaN values (use this if you want to exclude them)
    df_clean = df.dropna(subset=['label_vector'])
    print(f"Removed {len(df) - len(df_clean)} rows with NaN labels")
    df = df_clean
    

# Split the dataset into train (80%) and test (20%) sets
# We use stratify to maintain the same distribution of labels in both sets
train_df, test_df = train_test_split(
    df, 
    test_size=0.2,         # 20% for test set
    random_state=42,       # For reproducibility
    stratify=df['label_vector']  # Maintain label distribution
)

# Save the train and test sets to CSV files
train_df.to_csv(train_file, index=False)
test_df.to_csv(test_file, index=False)

print(f"Original dataset shape: {df.shape}")
print(f"Train dataset shape: {train_df.shape} ({len(train_df)/len(df)*100:.1f}%)")
print(f"Test dataset shape: {test_df.shape} ({len(test_df)/len(df)*100:.1f}%)")

# List of all 11 classes
classes = [
    "Broad Gender Bias",
    "Dismissive Addressing",
    "Everyday Derogation",
    "Fixed Gender Perceptions",
    "Harmful Provocation",
    "Hostile Speech",
    "Masked Disparagement",
    "Menacing Speech",
    "Singular Gender Bias",
    "Stripping Personhood",
    "Verbal Degradation"
]

# Print detailed class distribution
print("\n" + "="*50)
print("CLASS DISTRIBUTION IN DATASETS")
print("="*50)
print(f"{'Class':<25} {'Original':<10} {'Train':<10} {'Test':<10}")
print("-"*55)

for cls in classes:
    # Count examples in each dataset for this class
    orig_count = len(df[df['label_vector'] == cls])
    train_count = len(train_df[train_df['label_vector'] == cls])
    test_count = len(test_df[test_df['label_vector'] == cls])
    
    # Print counts with formatting
    print(f"{cls:<25} {orig_count:<10} {train_count:<10} {test_count:<10}")

# Print percentage distribution
print("\n" + "="*70)
print("CLASS DISTRIBUTION PERCENTAGES")
print("="*70)
print(f"{'Class':<25} {'Original %':<12} {'Train %':<12} {'Test %':<12}")
print("-"*70)

total_orig = len(df)
total_train = len(train_df)
total_test = len(test_df)

for cls in classes:
    # Calculate percentages
    orig_pct = len(df[df['label_vector'] == cls]) / total_orig * 100
    train_pct = len(train_df[train_df['label_vector'] == cls]) / total_train * 100
    test_pct = len(test_df[test_df['label_vector'] == cls]) / total_test * 100
    
    # Print percentages with formatting
    print(f"{cls:<25} {orig_pct:<12.2f} {train_pct:<12.2f} {test_pct:<12.2f}")

print("\nFiles saved successfully:")
print(f"- Training set: {train_file}")
print(f"- Test set: {test_file}")


