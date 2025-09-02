"""
Main script to run mouse dynamics classification pipeline
"""

import pandas as pd
import numpy as np
import os
from classifier import MouseDynamicsClassifier, analyze_features
from feature_extractor import load_session_data


def main():
    """
    Main pipeline for mouse dynamics classification
    """
    print("=== Mouse Dynamics Classification Pipeline ===\n")
    
    # Initialize classifier
    classifier = MouseDynamicsClassifier(random_state=42)
    
    # 1. Load and prepare training data
    print("Step 1: Loading training data...")
    X_train, y_train = classifier.load_data()
    
    if len(X_train) == 0:
        print("No training data found! Please check your data files.")
        return None, None, None
    
    print(f"Loaded {len(X_train)} training samples with {X_train.shape[1]} features")
    
    # 2. Train the model
    print("Step 2: Training the classifier...")
    training_results = classifier.train(X_train, y_train)
    
    print(f"Validation Accuracy: {training_results['validation_accuracy']:.4f}")
    print(f"Cross-validation Accuracy: {training_results['cv_mean_accuracy']:.4f} ± {training_results['cv_std_accuracy']:.4f}")
    
    # 3. Analyze features
    print("Step 3: Analyzing features...")
    print("Top 10 most important features:")
    print(training_results['feature_importance'].head(10))
    
    # 4. Load test data
    print("Step 4: Loading test data...")
    X_test, _, test_session_names = load_session_data('test_files')
    
    if len(X_test) == 0:
        print("No test data found!")
        return classifier, None, training_results
    
    print(f"Loaded {len(X_test)} test sessions")
    
    # 5. Make predictions
    print("Step 5: Making predictions...")
    test_predictions, test_probabilities = classifier.predict(X_test)
    
    print(f"Predicted {(test_predictions == 1).sum()} illegal sessions out of {len(test_predictions)} total")
    
    # 6. Create results DataFrame
    results_df = pd.DataFrame({
        'filename': test_session_names,
        'predicted_is_illegal': test_predictions,
        'illegal_probability': test_probabilities
    })
    
    # 7. Evaluate against public labels if available
    print("Step 6: Evaluating against public labels...")
    labels_df = pd.read_csv('public_labels.csv')
    public_results = results_df.merge(labels_df, on='filename', how='inner')
    
    if len(public_results) > 0:
        public_accuracy = (public_results['predicted_is_illegal'] == public_results['is_illegal']).mean()
        print(f"Accuracy on public test data: {public_accuracy:.4f}")
        print(f"Evaluated on {len(public_results)} public test sessions")
        
        # Show confusion matrix
        cm = training_results['confusion_matrix']
        print("Validation Confusion Matrix:")
        print("              Predicted")
        print("Actual    Legal  Illegal")
        print(f"Legal     {cm[0,0]:5d}  {cm[0,1]:7d}")
        print(f"Illegal   {cm[1,0]:5d}  {cm[1,1]:7d}")
    else:
        print("No matching sessions found between predictions and public labels")
    
    # 8. Save results
    print("Step 7: Saving results...")
    results_df.to_csv('predictions.csv', index=False)
    print("Predictions saved to 'predictions.csv'")
    
    if len(public_results) > 0:
        public_results.to_csv('public_evaluation.csv', index=False)
        print("Public evaluation saved to 'public_evaluation.csv'")
    
    # 9. Save trained model
    classifier.save_model('mouse_dynamics_model.pkl')
    
    # 10. Summary statistics
    print("=== SUMMARY ===")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features extracted: {X_train.shape[1]}")
    print(f"Validation accuracy: {training_results['validation_accuracy']:.4f}")
    if len(public_results) > 0:
        print(f"Public test accuracy: {public_accuracy:.4f}")
    print(f"Predicted illegal sessions: {(test_predictions == 1).sum()}/{len(test_predictions)}")
    
    return classifier, results_df, training_results


def analyze_data_distribution():
    """
    Analyze the distribution of data across users and sessions
    """
    print("=== Data Distribution Analysis ===\n")
    
    # Training data analysis
    print("Training data:")
    training_users = os.listdir('training_files')
    for user in training_users:
        user_path = os.path.join('training_files', user)
        session_count = len(os.listdir(user_path))
        print(f"  {user}: {session_count} sessions")
    
    total_training = sum(len(os.listdir(os.path.join('training_files', user))) 
                        for user in training_users)
    print(f"Total training sessions: {total_training}")
    
    # Test data analysis
    print("Test data:")
    test_users = os.listdir('test_files')
    for user in test_users:
        user_path = os.path.join('test_files', user)
        session_count = len(os.listdir(user_path))
        print(f"  {user}: {session_count} sessions")
    
    total_test = sum(len(os.listdir(os.path.join('test_files', user))) 
                    for user in test_users)
    print(f"Total test sessions: {total_test}")
    
    # Labels analysis
    labels_df = pd.read_csv('public_labels.csv')
    print(f"Public labels: {len(labels_df)} sessions")
    print(f"Legal: {(labels_df['is_illegal'] == 0).sum()}")
    print(f"Illegal: {(labels_df['is_illegal'] == 1).sum()}")


def generate_summary_report(classifier, results_df: pd.DataFrame, training_results: dict):
    """
    Generate a comprehensive summary report
    """
    labels_df = pd.read_csv('public_labels.csv')
    public_results = results_df.merge(labels_df, on='filename', how='inner')
    
    report = []
    report.append("=" * 60)
    report.append("MOUSE DYNAMICS CLASSIFICATION SUMMARY REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Model performance
    report.append("MODEL PERFORMANCE:")
    report.append(f"  Validation Accuracy: {training_results['validation_accuracy']:.4f}")
    report.append(f"  Cross-validation: {training_results['cv_mean_accuracy']:.4f} ± {training_results['cv_std_accuracy']:.4f}")
    
    if len(public_results) > 0:
        public_accuracy = (public_results['predicted_is_illegal'] == public_results['is_illegal']).mean()
        report.append(f"  Public Test Accuracy: {public_accuracy:.4f}")
    
    report.append("")
    
    # Data summary
    report.append("DATA SUMMARY:")
    report.append(f"  Training sessions processed: {len(classifier.feature_columns) if classifier.feature_columns is not None else 'N/A'}")
    report.append(f"  Test sessions processed: {len(results_df)}")
    report.append(f"  Features extracted: {len(training_results['feature_importance'])}")
    report.append("")
    
    # Prediction summary
    illegal_count = (results_df['predicted_is_illegal'] == 1).sum()
    legal_count = (results_df['predicted_is_illegal'] == 0).sum()
    
    report.append("PREDICTION SUMMARY:")
    report.append(f"  Predicted illegal sessions: {illegal_count}")
    report.append(f"  Predicted legal sessions: {legal_count}")
    report.append(f"  Illegal rate: {illegal_count/len(results_df)*100:.1f}%")
    report.append("")
    
    # Top features
    report.append("TOP 5 MOST IMPORTANT FEATURES:")
    for idx, row in training_results['feature_importance'].head(5).iterrows():
        report.append(f"  {row['feature']}: {row['importance']:.4f}")
    
    report.append("")
    report.append("=" * 60)
    
    # Save and print report
    report_text = "\n".join(report)
    
    with open('classification_report.txt', 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print("\nDetailed report saved to 'classification_report.txt'")


if __name__ == "__main__":
    # Run data analysis first
    analyze_data_distribution()
    
    print("\n" + "="*50 + "\n")
    
    # Run main classification pipeline
    classifier, results_df, training_results = main()
    
    if classifier is not None and results_df is not None:
        # Generate summary report
        generate_summary_report(classifier, results_df, training_results)
        
        print("\nPipeline completed successfully!")
        print("Generated files:")
        print("  - predictions.csv: Test predictions")
        print("  - public_evaluation.csv: Evaluation on public labels")
        print("  - mouse_dynamics_model.pkl: Trained model")
        print("  - classification_report.txt: Summary report")
    else:
        print("Pipeline failed. Check your data files and try again.")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_curve, auc
from typing import Dict, Any


def plot_training_results(training_results: Dict[str, Any], X_train: pd.DataFrame, y_train: np.ndarray):
    """
    Create comprehensive visualizations of training results
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Feature importance
    feature_importance = training_results['feature_importance']
    top_features = feature_importance.head(10)
    
    axes[0, 0].barh(top_features['feature'], top_features['importance'])
    axes[0, 0].set_title('Top 10 Feature Importance')
    axes[0, 0].set_xlabel('Importance')
    
    # Confusion matrix
    cm = training_results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
    axes[0, 1].set_title('Confusion Matrix (Validation)')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Actual')
    
    # Feature correlation heatmap
    correlation_matrix = X_train.corr()
    top_features_names = top_features['feature'].head(8).tolist()
    subset_corr = correlation_matrix.loc[top_features_names, top_features_names]
    
    sns.heatmap(subset_corr, annot=True, cmap='coolwarm', center=0, ax=axes[1, 0])
    axes[1, 0].set_title('Feature Correlation (Top 8 Features)')
    
    # Class distribution with key features
    key_feature = top_features.iloc[0]['feature']
    legal_data = X_train[y_train == 0][key_feature]
    illegal_data = X_train[y_train == 1][key_feature]
    
    axes[1, 1].hist(legal_data, alpha=0.5, label='Legal', bins=30, density=True)
    axes[1, 1].hist(illegal_data, alpha=0.5, label='Illegal', bins=30, density=True)
    axes[1, 1].set_title(f'Distribution of {key_feature}')
    axes[1, 1].set_xlabel(key_feature)
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('training_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_prediction_analysis(results_df: pd.DataFrame, labels_df: pd.DataFrame):
    """
    Analyze prediction results
    """
    # Merge with public labels
    public_results = results_df.merge(labels_df, on='filename', how='inner')
    
    if len(public_results) == 0:
        print("No public labels available for analysis")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Prediction confidence distribution
    axes[0, 0].hist(results_df['illegal_probability'], bins=50, alpha=0.7, density=True)
    axes[0, 0].set_title('Prediction Confidence Distribution')
    axes[0, 0].set_xlabel('Probability of Illegal Session')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].axvline(0.5, color='red', linestyle='--', label='Decision Threshold')
    axes[0, 0].legend()
    
    # ROC curve (if public labels available)
    if len(public_results) > 0:
        fpr, tpr, _ = roc_curve(public_results['is_illegal'], public_results['illegal_probability'])
        roc_auc = auc(fpr, tpr)
        
        axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend(loc="lower right")
        
        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(public_results['is_illegal'], public_results['illegal_probability'])
        pr_auc = auc(recall, precision)
        
        axes[1, 0].plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision-Recall Curve')
        axes[1, 0].legend()
        
        # Prediction vs actual scatter
        axes[1, 1].scatter(public_results['illegal_probability'], public_results['is_illegal'], alpha=0.6)
        axes[1, 1].set_xlabel('Predicted Probability')
        axes[1, 1].set_ylabel('Actual Label')
        axes[1, 1].set_title('Predicted Probability vs Actual Labels')
        axes[1, 1].set_yticks([0, 1])
        axes[1, 1].set_yticklabels(['Legal', 'Illegal'])
    
    plt.tight_layout()
    plt.savefig('prediction_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def generate_summary_report(classifier, results_df: pd.DataFrame, training_results: Dict[str, Any]):
    """
    Generate a comprehensive summary report
    """
    labels_df = pd.read_csv('public_labels.csv')
    public_results = results_df.merge(labels_df, on='filename', how='inner')
    
    report = []
    report.append("=" * 60)
    report.append("MOUSE DYNAMICS CLASSIFICATION SUMMARY REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Model performance
    report.append("MODEL PERFORMANCE:")
    report.append(f"  Validation Accuracy: {training_results['validation_accuracy']:.4f}")
    report.append(f"  Cross-validation: {training_results['cv_mean_accuracy']:.4f} ± {training_results['cv_std_accuracy']:.4f}")
    
    if len(public_results) > 0:
        public_accuracy = (public_results['predicted_is_illegal'] == public_results['is_illegal']).mean()
        report.append(f"  Public Test Accuracy: {public_accuracy:.4f}")
    
    report.append("")
    
    # Data summary
    report.append("DATA SUMMARY:")
    report.append(f"  Training sessions processed: {len(classifier.feature_columns) if classifier.feature_columns is not None else 'N/A'}")
    report.append(f"  Test sessions processed: {len(results_df)}")
    report.append(f"  Features extracted: {len(training_results['feature_importance'])}")
    report.append("")
    
    # Prediction summary
    illegal_count = (results_df['predicted_is_illegal'] == 1).sum()
    legal_count = (results_df['predicted_is_illegal'] == 0).sum()
    
    report.append("PREDICTION SUMMARY:")
    report.append(f"  Predicted illegal sessions: {illegal_count}")
    report.append(f"  Predicted legal sessions: {legal_count}")
    report.append(f"  Illegal rate: {illegal_count/len(results_df)*100:.1f}%")
    report.append("")
    
    # Top features
    report.append("TOP 5 MOST IMPORTANT FEATURES:")
    for idx, row in training_results['feature_importance'].head(5).iterrows():
        report.append(f"  {row['feature']}: {row['importance']:.4f}")
    
    report.append("")
    report.append("=" * 60)
    
    # Save and print report
    report_text = "\n".join(report)
    
    with open('classification_report.txt', 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print("\nDetailed report saved to 'classification_report.txt'")


if __name__ == "__main__":
    # Run the complete pipeline
    classifier, results_df, training_results = main()
    
    # Generate comprehensive analysis
    print("\n" + "="*50)
    print("GENERATING ANALYSIS PLOTS...")
    
    try:
        # Load training data for analysis
        X_train, y_train = classifier.load_data()
        analyze_features(X_train, y_train, training_results['feature_importance'])
        
        labels_df = pd.read_csv('public_labels.csv')
        plot_prediction_analysis(results_df, labels_df)
        
    except ImportError as e:
        print(f"Skipping visualizations due to missing dependencies: {e}")
        print("Install matplotlib and seaborn for visualizations")
    
    # Generate summary report
    generate_summary_report(classifier, results_df, training_results)
    
    print("\nAll analysis complete! Check the generated files:")
    print("  - predictions.csv: Test predictions")
    print("  - public_evaluation.csv: Evaluation on public labels")
    print("  - mouse_dynamics_model.pkl: Trained model")
    print("  - classification_report.txt: Summary report")
    print("  - training_analysis.png: Feature analysis plots")
    print("  - prediction_analysis.png: Prediction analysis plots")
