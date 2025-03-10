"""
Toxicity Dataset Analysis Script

This script performs basic analysis of the Google Jigsaw toxicity dataset, focusing on:
- Label distribution and imbalance
- Label correlation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style for plots
plt.style.use('ggplot')
sns.set(font_scale=1.2)

def load_data(data_path):
    """Load dataset from CSV file"""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")
    return df

def examine_structure(df):
    """Examine dataset structure and missing values"""
    print("\n=== Dataset Structure ===")
    print(f"Columns: {df.columns.tolist()}")
    print("\n=== Data Types ===")
    print(df.dtypes)
    
    print("\n=== Missing Values ===")
    missing = df.isnull().sum()
    print(missing[missing > 0])
    
    print("\n=== Sample Data ===")
    print(df.head(3))
    
    return df

def analyze_labels(df):
    """Analyze label distribution"""
    print("\n=== Label Distribution ===")
    
    # Assuming 'toxic' is the main label column - adjust accordingly
    # Check if we have a multi-label or binary classification problem
    label_columns = [col for col in df.columns if col in 
                    ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]
    
    if label_columns:
        print(f"Label columns found: {label_columns}")
        
        # Count distribution for each label
        for col in label_columns:
            count = df[col].value_counts()
            print(f"\n{col.upper()}")
            print(count)
            print(f"Percentage positive: {count[1] / count.sum() * 100:.2f}%")
            
        # Create a plot for label distribution
        plt.figure(figsize=(12, 6))
        pos_counts = [df[col].sum() for col in label_columns]
        neg_counts = [df.shape[0] - count for count in pos_counts]
        
        x = np.arange(len(label_columns))
        width = 0.35
        
        plt.bar(x - width/2, pos_counts, width, label='Positive')
        plt.bar(x + width/2, neg_counts, width, label='Negative')
        plt.xticks(x, [col.replace('_', ' ').title() for col in label_columns])
        plt.ylabel('Count')
        plt.yscale('log')
        plt.title('Label Distribution (Log Scale)')
        plt.legend()
        
        # Save the plot
        os.makedirs('../plots', exist_ok=True)
        plt.savefig('../plots/label_distribution.png')
        plt.close()
        
        # Look for co-occurrence of labels
        if len(label_columns) > 1:
            print("\n=== Label Co-occurrence ===")
            corr = df[label_columns].corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title('Label Correlation')
            plt.tight_layout()
            plt.savefig('../plots/label_correlation.png')
            plt.close()
            
    else:
        print("No standard toxicity label columns found. Please adjust the script for your dataset structure.")
    
    return df

def merge_test_with_labels(test_df, test_labels_df):
    """Merge test data with its labels"""
    return pd.merge(test_df, test_labels_df, on='id', how='inner')

def main():
    """Main function to run all analyses"""
    # Data paths
    train_path = "../data/train.csv"
    test_path = "../data/test.csv"
    test_labels_path = "../data/test_labels.csv"
    
    try:
        # Check if files exist
        for path in [train_path, test_path, test_labels_path]:
            if not os.path.exists(path):
                print(f"Error: File {path} not found. Please update the path.")
                return
        
        # Create plots directory
        os.makedirs('../plots', exist_ok=True)
        
        # Load data
        print("Loading datasets...")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        test_labels_df = pd.read_csv(test_labels_path)
        
        # Merge test data with its labels
        full_test_df = merge_test_with_labels(test_df, test_labels_df)
        
        # Combine train and test for overall analysis
        print("Combining datasets for analysis...")
        combined_df = pd.concat([train_df, full_test_df], ignore_index=True)
        
        # Analyze combined data
        print("\n=== COMBINED DATASET ANALYSIS ===")
        combined_df = examine_structure(combined_df)
        combined_df = analyze_labels(combined_df)
        
        print("\nAnalysis complete! Plots saved to the '../plots' directory.")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()