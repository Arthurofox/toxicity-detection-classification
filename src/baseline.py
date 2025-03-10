"""
LLM Baseline for Toxicity Classification

This script creates a baseline using a pre-trained LLM to classify toxic content.
It loads a sample of test data and runs it through the LLM for classification.
"""

import pandas as pd
import numpy as np
import torch
import json
import os
from tqdm import tqdm
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import argparse
from sklearn.metrics import classification_report, f1_score, roc_auc_score

# Label names for reporting
LABEL_NAMES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def load_test_sample(test_path="../data/test.csv", 
                    test_labels_path="../data/test_labels.csv",
                    sample_indices_path="../data/sample_test_indices.npy"):
    """
    Load sample test data using saved indices for reproducibility
    """
    print("Loading test data sample...")
    
    # Load test data and labels
    test_df = pd.read_csv(test_path)
    test_labels_df = pd.read_csv(test_labels_path)
    
    # Merge test data with its labels
    test_df = pd.merge(test_df, test_labels_df, on='id', how='inner')
    
    # Remove rows with -1 in test data (unlabeled examples)
    test_df = test_df[test_df['toxic'] != -1].reset_index(drop=True)
    
    # Load sample indices
    if os.path.exists(sample_indices_path):
        sample_indices = np.load(sample_indices_path)
        test_sample = test_df.iloc[sample_indices].reset_index(drop=True)
        print(f"Loaded {len(test_sample)} test samples using saved indices")
    else:
        # If no sample indices, use all data
        test_sample = test_df
        print("No sample indices found, using all test data")
    
    return test_sample

def run_baseline_zero_shot(test_sample, model_name="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", 
                          device="cpu", batch_size=8):
    """
    Run zero-shot classification using a pre-trained model
    """
    print(f"Running zero-shot classification with {model_name}...")
    
    # Load model and tokenizer
    classifier = pipeline(
        "zero-shot-classification",
        model=model_name,
        device=0 if device == "cuda" and torch.cuda.is_available() else -1,
        batch_size=batch_size
    )
    
    # Define candidate labels for toxicity
    candidate_labels = ["toxic", "not toxic"]
    
    # Process test samples in batches
    texts = test_sample['comment_text'].tolist()
    results = []
    
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        batch_results = classifier(batch_texts, candidate_labels, multi_label=False)
        results.extend(batch_results)
    
    # Extract predictions and scores
    predictions = []
    scores = []
    
    for result in results:
        # Get the index of "toxic" in the labels
        toxic_idx = result['labels'].index("toxic")
        # Get the score for "toxic"
        toxic_score = result['scores'][toxic_idx]
        # Predict toxic if score > 0.5
        prediction = 1 if toxic_score > 0.5 else 0
        
        predictions.append(prediction)
        scores.append(toxic_score)
    
    return predictions, scores

def run_baseline_text_classification(test_sample, model_name="unitary/toxic-bert", device="cpu", batch_size=8):
    """
    Run text classification using a pre-trained toxic content classifier
    """
    print(f"Running text classification with {model_name}...")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    if device == "cuda" and torch.cuda.is_available():
        model = model.to("cuda")
    
    # Process test samples in batches
    texts = test_sample['comment_text'].tolist()
    all_predictions = []
    all_scores = []
    
    # Check if multi-label or single-label
    multi_label = model.config.problem_type == "multi_label_classification"
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
            if device == "cuda" and torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Run inference
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Convert logits to predictions and scores
            if multi_label:
                # Multi-label: use sigmoid
                scores = torch.sigmoid(logits).cpu().numpy()
                preds = (scores > 0.5).astype(int)
            else:
                # Single-label: use softmax
                scores = torch.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(scores, axis=1)
                
                # If binary classification, extract positive class score
                if scores.shape[1] == 2:
                    scores = scores[:, 1]  # Positive class score
            
            all_predictions.extend(preds)
            all_scores.extend(scores)
    
    # Convert to numpy arrays
    if multi_label:
        all_predictions = np.array(all_predictions)
        all_scores = np.array(all_scores)
    else:
        all_predictions = np.array(all_predictions)
        all_scores = np.array(all_scores)
        # For binary classification, reshape scores if needed
        if len(all_scores.shape) == 1:
            all_scores = all_scores.reshape(-1, 1)
    
    return all_predictions, all_scores

def evaluate_baseline(predictions, scores, true_labels, multi_label=False):
    """
    Evaluate baseline performance
    """
    print("\n=== Baseline Model Performance ===")
    
    # Convert predictions to numpy arrays if they aren't already
    if isinstance(predictions, list):
        predictions = np.array(predictions)
    if isinstance(scores, list):
        scores = np.array(scores)
    
    # For multi-label evaluation
    if multi_label:
        # Calculate metrics for each label
        for i, label in enumerate(LABEL_NAMES):
            y_true = true_labels[:, i]
            y_pred = predictions[:, i] if len(predictions.shape) > 1 else predictions
            y_score = scores[:, i] if len(scores.shape) > 1 else scores
            
            print(f"\n--- {label} ---")
            print(classification_report(y_true, y_pred))
            
            try:
                # Calculate ROC AUC
                auc = roc_auc_score(y_true, y_score)
                print(f"ROC AUC: {auc:.4f}")
            except Exception as e:
                print(f"Could not calculate ROC AUC: {e}")
        
        # Calculate micro and macro F1
        micro_f1 = f1_score(true_labels, predictions, average='micro')
        macro_f1 = f1_score(true_labels, predictions, average='macro')
        
        print("\n--- Overall ---")
        print(f"Micro F1: {micro_f1:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
    
    # For binary classification (primary toxic label only)
    else:
        # Use only the first column (toxic) for evaluation
        y_true = true_labels[:, 0]
        
        print(classification_report(y_true, predictions))
        
        try:
            # Calculate ROC AUC
            auc = roc_auc_score(y_true, scores)
            print(f"ROC AUC: {auc:.4f}")
        except Exception as e:
            print(f"Could not calculate ROC AUC: {e}")
    
    # Save results
    results = {
        "model_type": "baseline",
        "predictions": predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
        "scores": scores.tolist() if isinstance(scores, np.ndarray) else scores,
        "metrics": {
            "micro_f1": f1_score(true_labels, predictions, average='micro') if multi_label else f1_score(true_labels[:, 0], predictions),
            "macro_f1": f1_score(true_labels, predictions, average='macro') if multi_label else None,
        }
    }
    
    # Save to file
    os.makedirs("../results", exist_ok=True)
    with open("../results/baseline_results.json", "w") as f:
        json.dump(results, f)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run LLM baseline for toxicity classification")
    parser.add_argument("--model", type=str, default="unitary/toxic-bert", 
                        help="Model name or path")
    parser.add_argument("--method", type=str, choices=["zero-shot", "classification"], 
                        default="classification", help="Classification method")
    parser.add_argument("--batch-size", type=int, default=8, 
                        help="Batch size for inference")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], 
                        default="cpu", help="Device for inference")
    parser.add_argument("--multi-label", action="store_true", 
                        help="Whether to perform multi-label classification")
    
    args = parser.parse_args()
    
    # Load test sample
    test_sample = load_test_sample()
    
    # Extract true labels
    true_labels = test_sample[LABEL_NAMES].values
    
    # Run baseline
    if args.method == "zero-shot":
        predictions, scores = run_baseline_zero_shot(
            test_sample, 
            model_name=args.model,
            device=args.device,
            batch_size=args.batch_size
        )
    else:
        predictions, scores = run_baseline_text_classification(
            test_sample,
            model_name=args.model,
            device=args.device,
            batch_size=args.batch_size
        )
    
    # Evaluate baseline
    results = evaluate_baseline(predictions, scores, true_labels, args.multi_label)
    
    print(f"Baseline results saved to ../results/baseline_results.json")

if __name__ == "__main__":
    main()