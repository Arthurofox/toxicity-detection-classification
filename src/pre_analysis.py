"""
Toxicity Dataset Analysis Script

This script performs a basic analysis of the Google Jigsaw toxicity dataset to understand:
- Label distribution and imbalance
- Text length distribution
- Word frequency analysis
- Basic demographic analysis (if available)
- Sample toxic and non-toxic examples
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import os
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

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
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/label_distribution.png')
        plt.close()
        
        # Look for co-occurrence of labels
        if len(label_columns) > 1:
            print("\n=== Label Co-occurrence ===")
            corr = df[label_columns].corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title('Label Correlation')
            plt.tight_layout()
            plt.savefig('plots/label_correlation.png')
            plt.close()
            
    else:
        print("No standard toxicity label columns found. Please adjust the script for your dataset structure.")
    
    return df

def analyze_text_length(df, text_column='comment_text'):
    """Analyze text length distribution"""
    if text_column not in df.columns:
        print(f"Warning: '{text_column}' column not found. Please specify the correct text column.")
        return df
    
    print(f"\n=== Text Length Analysis ({text_column}) ===")
    
    # Add character and word count columns
    df['char_count'] = df[text_column].str.len()
    df['word_count'] = df[text_column].str.split().str.len()
    
    # Basic statistics
    print("\nCharacter count statistics:")
    print(df['char_count'].describe())
    
    print("\nWord count statistics:")
    print(df['word_count'].describe())
    
    # Create plots for text length distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Character count
    sns.histplot(data=df, x='char_count', hue='toxic' if 'toxic' in df.columns else None, 
                bins=50, kde=True, ax=ax1)
    ax1.set_title('Character Count Distribution')
    ax1.set_xlabel('Character Count')
    ax1.set_ylabel('Frequency')
    
    # Word count
    sns.histplot(data=df, x='word_count', hue='toxic' if 'toxic' in df.columns else None, 
                bins=50, kde=True, ax=ax2)
    ax2.set_title('Word Count Distribution')
    ax2.set_xlabel('Word Count')
    ax2.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('plots/text_length_distribution.png')
    plt.close()
    
    # Check for significant differences in length between toxic and non-toxic
    if 'toxic' in df.columns:
        print("\nAverage text length by toxicity:")
        print(df.groupby('toxic')['char_count'].mean())
        print(df.groupby('toxic')['word_count'].mean())
    
    return df

def analyze_vocabulary(df, text_column='comment_text', max_features=100):
    """Analyze word frequency and vocabulary"""
    if text_column not in df.columns:
        print(f"Warning: '{text_column}' column not found.")
        return df
    
    print("\n=== Vocabulary Analysis ===")
    
    # Create word clouds for toxic and non-toxic content
    if 'toxic' in df.columns:
        # Function to clean text
        def clean_text(text):
            if isinstance(text, str):
                # Convert to lowercase and remove punctuation
                text = re.sub(r'[^\w\s]', '', text.lower())
                return text
            return ""
        
        df['clean_text'] = df[text_column].apply(clean_text)
        
        # Word cloud for toxic content
        toxic_text = ' '.join(df[df['toxic'] == 1]['clean_text'])
        if toxic_text.strip():
            toxic_cloud = WordCloud(width=800, height=400, background_color='white', 
                                   max_words=100, contour_width=3).generate(toxic_text)
            
            plt.figure(figsize=(10, 8))
            plt.imshow(toxic_cloud, interpolation='bilinear')
            plt.title('Word Cloud - Toxic Content')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig('plots/toxic_wordcloud.png')
            plt.close()
        
        # Word cloud for non-toxic content
        nontoxic_text = ' '.join(df[df['toxic'] == 0]['clean_text'])
        if nontoxic_text.strip():
            nontoxic_cloud = WordCloud(width=800, height=400, background_color='white', 
                                      max_words=100, contour_width=3).generate(nontoxic_text)
            
            plt.figure(figsize=(10, 8))
            plt.imshow(nontoxic_cloud, interpolation='bilinear')
            plt.title('Word Cloud - Non-Toxic Content')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig('plots/nontoxic_wordcloud.png')
            plt.close()
    
    # Use CountVectorizer to get most common words
    vectorizer = CountVectorizer(max_features=max_features, stop_words='english')
    X = vectorizer.fit_transform(df[text_column].fillna(''))
    
    word_freq = dict(zip(vectorizer.get_feature_names_out(), X.sum(axis=0).tolist()[0]))
    word_freq = {k: v for k, v in sorted(word_freq.items(), key=lambda item: item[1], reverse=True)}
    
    print(f"\nTop {min(20, len(word_freq))} words overall:")
    for word, count in list(word_freq.items())[:20]:
        print(f"{word}: {count}")
    
    # Plot word frequencies
    plt.figure(figsize=(12, 8))
    words = list(word_freq.keys())[:20]
    counts = list(word_freq.values())[:20]
    
    sns.barplot(x=counts, y=words)
    plt.title('Top 20 Words by Frequency')
    plt.xlabel('Frequency')
    plt.tight_layout()
    plt.savefig('plots/word_frequency.png')
    plt.close()
    
    return df

def analyze_demographics(df):
    """Analyze demographic information if available"""
    demographic_cols = [col for col in df.columns if col.lower() in 
                       ['gender', 'race', 'age', 'religion', 'sexual_orientation', 'disability']]
    
    if demographic_cols:
        print("\n=== Demographic Analysis ===")
        print(f"Demographic columns found: {demographic_cols}")
        
        for col in demographic_cols:
            print(f"\n{col.upper()} Distribution:")
            value_counts = df[col].value_counts()
            print(value_counts)
            
            # Create plot
            plt.figure(figsize=(10, 6))
            value_counts.plot(kind='bar')
            plt.title(f'{col.title()} Distribution')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'plots/{col}_distribution.png')
            plt.close()
            
            # Check toxicity rates by demographic group if 'toxic' column exists
            if 'toxic' in df.columns:
                print(f"\nToxicity rate by {col}:")
                toxicity_by_group = df.groupby(col)['toxic'].mean()
                print(toxicity_by_group)
                
                plt.figure(figsize=(10, 6))
                toxicity_by_group.plot(kind='bar')
                plt.title(f'Toxicity Rate by {col.title()}')
                plt.ylabel('Toxicity Rate')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f'plots/toxicity_by_{col}.png')
                plt.close()
    else:
        print("\nNo demographic columns found in the dataset.")
    
    return df

def sample_examples(df, text_column='comment_text', n=5):
    """Print sample examples of toxic and non-toxic content"""
    if 'toxic' not in df.columns or text_column not in df.columns:
        print("Cannot sample examples: missing required columns.")
        return
    
    print("\n=== Sample Examples ===")
    
    # Sample toxic examples
    print("\nTOXIC EXAMPLES:")
    toxic_samples = df[df['toxic'] == 1].sample(n=min(n, df[df['toxic'] == 1].shape[0]))
    for i, (_, row) in enumerate(toxic_samples.iterrows(), 1):
        print(f"{i}. {row[text_column][:200]}{'...' if len(row[text_column]) > 200 else ''}")
    
    # Sample non-toxic examples
    print("\nNON-TOXIC EXAMPLES:")
    nontoxic_samples = df[df['toxic'] == 0].sample(n=min(n, df[df['toxic'] == 0].shape[0]))
    for i, (_, row) in enumerate(nontoxic_samples.iterrows(), 1):
        print(f"{i}. {row[text_column][:200]}{'...' if len(row[text_column]) > 200 else ''}")

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
        os.makedirs('plots', exist_ok=True)
        
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
        combined_df = analyze_text_length(combined_df)
        combined_df = analyze_vocabulary(combined_df)
        combined_df = analyze_demographics(combined_df)
        sample_examples(combined_df)
        
        # Also analyze training data separately
        print("\n\n=== TRAINING DATASET ANALYSIS ===")
        train_df = examine_structure(train_df)
        train_df = analyze_labels(train_df)
        train_df = analyze_text_length(train_df)
        
        # Analyze test data separately
        print("\n\n=== TEST DATASET ANALYSIS ===")
        full_test_df = examine_structure(full_test_df)
        full_test_df = analyze_labels(full_test_df)
        full_test_df = analyze_text_length(full_test_df)
        
        print("\nAnalysis complete! Plots saved to the 'plots' directory.")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()