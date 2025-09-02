"""
Performance analysis script for mouse dynamics classification model
"""

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def analyze_model_performance():
    """
    Analyze how well our model predictions match the actual truth
    """
    # Load the public evaluation results
    df = pd.read_csv('public_evaluation.csv')
    
    print('=' * 60)
    print('MOUSE DYNAMICS MODEL PERFORMANCE ANALYSIS')
    print('=' * 60)
    print(f'Total evaluated sessions: {len(df)}')
    print()
    
    # Calculate overall accuracy
    correct = (df['predicted_is_illegal'] == df['is_illegal']).sum()
    accuracy = correct / len(df)
    print(f'Overall Accuracy: {accuracy:.4f} ({correct}/{len(df)})')
    print()
    
    # Detailed breakdown by class
    legal_actual = df[df['is_illegal'] == 0]
    illegal_actual = df[df['is_illegal'] == 1]
    
    print('=' * 40)
    print('LEGAL SESSIONS ANALYSIS (is_illegal = 0)')
    print('=' * 40)
    legal_correct = (legal_actual['predicted_is_illegal'] == 0).sum()
    legal_wrong = (legal_actual['predicted_is_illegal'] == 1).sum()
    print(f'Total actual legal sessions: {len(legal_actual)}')
    print(f'Correctly identified as legal: {legal_correct} ({legal_correct/len(legal_actual):.4f})')
    print(f'Incorrectly identified as illegal: {legal_wrong} ({legal_wrong/len(legal_actual):.4f})')
    print()
    
    print('=' * 40)
    print('ILLEGAL SESSIONS ANALYSIS (is_illegal = 1)')
    print('=' * 40)
    illegal_correct = (illegal_actual['predicted_is_illegal'] == 1).sum()
    illegal_wrong = (illegal_actual['predicted_is_illegal'] == 0).sum()
    print(f'Total actual illegal sessions: {len(illegal_actual)}')
    print(f'Correctly identified as illegal: {illegal_correct} ({illegal_correct/len(illegal_actual):.4f})')
    print(f'Incorrectly identified as legal: {illegal_wrong} ({illegal_wrong/len(illegal_actual):.4f})')
    print()
    
    # Confusion Matrix
    print('=' * 40)
    print('CONFUSION MATRIX')
    print('=' * 40)
    print('                Predicted')
    print('Actual     Legal  Illegal  Total')
    legal_pred_legal = ((df['is_illegal'] == 0) & (df['predicted_is_illegal'] == 0)).sum()
    legal_pred_illegal = ((df['is_illegal'] == 0) & (df['predicted_is_illegal'] == 1)).sum()
    illegal_pred_legal = ((df['is_illegal'] == 1) & (df['predicted_is_illegal'] == 0)).sum()
    illegal_pred_illegal = ((df['is_illegal'] == 1) & (df['predicted_is_illegal'] == 1)).sum()
    
    print(f'Legal      {legal_pred_legal:5d}   {legal_pred_illegal:7d}   {len(legal_actual):5d}')
    print(f'Illegal    {illegal_pred_legal:5d}   {illegal_pred_illegal:7d}   {len(illegal_actual):5d}')
    print(f'Total      {legal_pred_legal + illegal_pred_legal:5d}   {legal_pred_illegal + illegal_pred_illegal:7d}   {len(df):5d}')
    print()
    
    # Calculate precision, recall, F1 scores
    print('=' * 40)
    print('DETAILED PERFORMANCE METRICS')
    print('=' * 40)
    
    # For Legal class (0)
    precision_legal = legal_pred_legal / (legal_pred_legal + illegal_pred_legal) if (legal_pred_legal + illegal_pred_legal) > 0 else 0
    recall_legal = legal_pred_legal / len(legal_actual) if len(legal_actual) > 0 else 0
    f1_legal = 2 * (precision_legal * recall_legal) / (precision_legal + recall_legal) if (precision_legal + recall_legal) > 0 else 0
    
    # For Illegal class (1)
    precision_illegal = illegal_pred_illegal / (illegal_pred_illegal + legal_pred_illegal) if (illegal_pred_illegal + legal_pred_illegal) > 0 else 0
    recall_illegal = illegal_pred_illegal / len(illegal_actual) if len(illegal_actual) > 0 else 0
    f1_illegal = 2 * (precision_illegal * recall_illegal) / (precision_illegal + recall_illegal) if (precision_illegal + recall_illegal) > 0 else 0
    
    print('Legal Sessions (Class 0):')
    print(f'  Precision: {precision_legal:.4f} (of predicted legal, how many were actually legal)')
    print(f'  Recall:    {recall_legal:.4f} (of actual legal, how many were correctly identified)')
    print(f'  F1-Score:  {f1_legal:.4f}')
    print()
    print('Illegal Sessions (Class 1):')
    print(f'  Precision: {precision_illegal:.4f} (of predicted illegal, how many were actually illegal)')
    print(f'  Recall:    {recall_illegal:.4f} (of actual illegal, how many were correctly identified)')
    print(f'  F1-Score:  {f1_illegal:.4f}')
    print()
    
    # Error analysis
    print('=' * 40)
    print('ERROR ANALYSIS')
    print('=' * 40)
    
    # False positives (predicted illegal, actually legal)
    false_positives = df[(df['is_illegal'] == 0) & (df['predicted_is_illegal'] == 1)]
    print(f'False Positives: {len(false_positives)} sessions')
    if len(false_positives) > 0:
        print(f'  Average confidence: {false_positives["illegal_probability"].mean():.4f}')
        print(f'  Confidence range: {false_positives["illegal_probability"].min():.4f} - {false_positives["illegal_probability"].max():.4f}')
    
    # False negatives (predicted legal, actually illegal)
    false_negatives = df[(df['is_illegal'] == 1) & (df['predicted_is_illegal'] == 0)]
    print(f'False Negatives: {len(false_negatives)} sessions')
    if len(false_negatives) > 0:
        print(f'  Average confidence: {false_negatives["illegal_probability"].mean():.4f}')
        print(f'  Confidence range: {false_negatives["illegal_probability"].min():.4f} - {false_negatives["illegal_probability"].max():.4f}')
    print()
    
    # Confidence analysis
    print('=' * 40)
    print('PREDICTION CONFIDENCE ANALYSIS')
    print('=' * 40)
    
    # High confidence correct predictions
    high_conf_correct = df[
        ((df['illegal_probability'] > 0.8) & (df['predicted_is_illegal'] == df['is_illegal'])) |
        ((df['illegal_probability'] < 0.2) & (df['predicted_is_illegal'] == df['is_illegal']))
    ]
    
    print(f'High confidence correct predictions: {len(high_conf_correct)}/{len(df)} ({len(high_conf_correct)/len(df):.4f})')
    
    # Low confidence predictions (near 0.5)
    low_conf = df[(df['illegal_probability'] > 0.4) & (df['illegal_probability'] < 0.6)]
    print(f'Low confidence predictions (0.4-0.6): {len(low_conf)}/{len(df)} ({len(low_conf)/len(df):.4f})')
    if len(low_conf) > 0:
        low_conf_accuracy = (low_conf['predicted_is_illegal'] == low_conf['is_illegal']).mean()
        print(f'  Accuracy on low confidence predictions: {low_conf_accuracy:.4f}')
    
    print()
    
    # Sample misclassifications
    print('=' * 40)
    print('SAMPLE MISCLASSIFICATIONS')
    print('=' * 40)
    
    misclassified = df[df['predicted_is_illegal'] != df['is_illegal']]
    if len(misclassified) > 0:
        print('Sample False Positives (predicted illegal, actually legal):')
        fp_sample = false_positives.head(3)
        for _, row in fp_sample.iterrows():
            print(f'  {row["filename"]}: confidence = {row["illegal_probability"]:.4f}')
        
        print()
        print('Sample False Negatives (predicted legal, actually illegal):')
        fn_sample = false_negatives.head(3)
        for _, row in fn_sample.iterrows():
            print(f'  {row["filename"]}: confidence = {row["illegal_probability"]:.4f}')
    
    print()
    
    # Probability distribution analysis
    print('=' * 40)
    print('PROBABILITY DISTRIBUTION ANALYSIS')
    print('=' * 40)
    
    legal_probs = df[df['is_illegal'] == 0]['illegal_probability']
    illegal_probs = df[df['is_illegal'] == 1]['illegal_probability']
    
    print('Legal sessions probability statistics:')
    print(f'  Mean: {legal_probs.mean():.4f}')
    print(f'  Std:  {legal_probs.std():.4f}')
    print(f'  Min:  {legal_probs.min():.4f}')
    print(f'  Max:  {legal_probs.max():.4f}')
    
    print()
    print('Illegal sessions probability statistics:')
    print(f'  Mean: {illegal_probs.mean():.4f}')
    print(f'  Std:  {illegal_probs.std():.4f}')
    print(f'  Min:  {illegal_probs.min():.4f}')
    print(f'  Max:  {illegal_probs.max():.4f}')
    
    print()
    print('=' * 60)
    
    return {
        'accuracy': accuracy,
        'precision_legal': precision_legal,
        'recall_legal': recall_legal,
        'f1_legal': f1_legal,
        'precision_illegal': precision_illegal,
        'recall_illegal': recall_illegal,
        'f1_illegal': f1_illegal,
        'false_positives': len(false_positives),
        'false_negatives': len(false_negatives)
    }


def save_analysis_report():
    """
    Save detailed analysis to a text file
    """
    import sys
    from io import StringIO
    
    # Capture the output
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()
    
    # Run analysis
    metrics = analyze_model_performance()
    
    # Get the captured output
    analysis_output = captured_output.getvalue()
    sys.stdout = old_stdout
    
    # Save to file
    with open('detailed_performance_analysis.txt', 'w') as f:
        f.write(analysis_output)
    
    print("Detailed performance analysis saved to 'detailed_performance_analysis.txt'")
    print(analysis_output)
    
    return metrics


if __name__ == '__main__':
    metrics = save_analysis_report()
