import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer
import torch
from torch.utils.data import Dataset, DataLoader

class ToxicityDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Convert to PyTorch tensors
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        
        # Add labels if available
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
            
        return item

def clean_text(text):
    """Basic text cleaning while preserving toxicity markers"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace URLs with token
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
    
    # Replace emails with token
    text = re.sub(r'\S+@\S+', '[EMAIL]', text)
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def prepare_data(train_path="../data/train.csv", 
                test_path="../data/test.csv", 
                test_labels_path="../data/test_labels.csv",
                val_size=0.1,
                random_state=42,
                tokenizer_name="roberta-base",
                max_length=256,
                batch_size=16,
                sample_test=None):
    """
    Prepare data for training, validation and testing
    
    Args:
        train_path: Path to training data
        test_path: Path to test data
        test_labels_path: Path to test labels
        val_size: Proportion of training data to use for validation
        random_state: Random seed for reproducibility
        tokenizer_name: Name of the tokenizer to use
        max_length: Maximum sequence length
        batch_size: Batch size for DataLoader
        sample_test: Number of test samples to use (None for all)
    
    Returns:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        class_weights: Weights for each class (for handling imbalance)
    """
    print("Loading datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    test_labels_df = pd.read_csv(test_labels_path)
    
    # Merge test data with its labels
    test_df = pd.merge(test_df, test_labels_df, on='id', how='inner')
    
    # Remove rows with -1 in test data (unlabeled examples)
    test_df = test_df[test_df['toxic'] != -1].reset_index(drop=True)
    
    # Clean text
    print("Cleaning text...")
    train_df['cleaned_text'] = train_df['comment_text'].apply(clean_text)
    test_df['cleaned_text'] = test_df['comment_text'].apply(clean_text)
    
    # Create multi-label and binary targets
    label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # Create train/validation split with stratification
    print("Creating train/validation split...")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_df['cleaned_text'].values,
        train_df[label_columns].values,
        test_size=val_size,
        random_state=random_state,
        stratify=train_df['toxic']  # Stratify by main toxic label
    )
    
    # Sample test data if requested
    if sample_test is not None:
        np.random.seed(random_state)
        sample_indices = np.random.choice(len(test_df), min(sample_test, len(test_df)), replace=False)
        test_texts = test_df['cleaned_text'].values[sample_indices]
        test_labels = test_df[label_columns].values[sample_indices]
    else:
        test_texts = test_df['cleaned_text'].values
        test_labels = test_df[label_columns].values
    
    # Load tokenizer
    print(f"Loading {tokenizer_name} tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
    
    # Create datasets
    train_dataset = ToxicityDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = ToxicityDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = ToxicityDataset(test_texts, test_labels, tokenizer, max_length)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Calculate class weights for handling imbalance
    class_weights = []
    for col in label_columns:
        # Calculate weight as inverse of class frequency
        neg_weight = 1.0 / (1 - train_df[col].mean())
        pos_weight = 1.0 / train_df[col].mean()
        class_weights.append(pos_weight / neg_weight)
    
    print(f"Created datasets with {len(train_loader)} training batches, {len(val_loader)} validation batches, and {len(test_loader)} test batches")
    
    # Save sample test indices if sampling was done
    if sample_test is not None:
        np.save("../data/sample_test_indices.npy", sample_indices)
        print(f"Saved {len(sample_indices)} test sample indices for reproducibility")
    
    # Return torch tensors of class weights
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    
    return train_loader, val_loader, test_loader, class_weights

if __name__ == "__main__":
    # Example usage
    train_loader, val_loader, test_loader, class_weights = prepare_data(
        sample_test=1000  # Sample 1000 test examples
    )
    
    print("Class weights for handling imbalance:", class_weights)
    
    # Print example batch
    for batch in train_loader:
        print("Example batch:")
        print(f"Input shape: {batch['input_ids'].shape}")
        print(f"Attention mask shape: {batch['attention_mask'].shape}")
        print(f"Labels shape: {batch['labels'].shape}")
        break