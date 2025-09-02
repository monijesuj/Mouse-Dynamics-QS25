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
        return
    
    print(f"Loaded {len(X_train)} training samples with {X_train.shape[1]} features")
    
    # 2. Train the model
    print("\nStep 2: Training the classifier...")
    training_results = classifier.train(X_train, y_train)
    
    print(f"Validation Accuracy: {training_results['validation_accuracy']:.4f}")
    print(f"Cross-validation Accuracy: {training_results['cv_mean_accuracy']:.4f} ± {training_results['cv_std_accuracy']:.4f}")
    
    # 3. Analyze features
    print("\nStep 3: Analyzing features...")
    print("Top 10 most important features:")
    print(training_results['feature_importance'].head(10))
    
    # 4. Load test data
    print("\nStep 4: Loading test data...")
    X_test, _, test_session_names = load_session_data('test_files')
    
    if len(X_test) == 0:
        print("No test data found!")
        return
    
    print(f"Loaded {len(X_test)} test sessions")
    
    # 5. Make predictions
    print("\nStep 5: Making predictions...")
    test_predictions, test_probabilities = classifier.predict(X_test)
    
    print(f"Predicted {(test_predictions == 1).sum()} illegal sessions out of {len(test_predictions)} total")
    
    # 6. Create results DataFrame
    results_df = pd.DataFrame({
        'filename': test_session_names,
        'predicted_is_illegal': test_predictions,
        'illegal_probability': test_probabilities
    })
    
    # 7. Evaluate against public labels if available
    print("\nStep 6: Evaluating against public labels...")
    labels_df = pd.read_csv('public_labels.csv')
    public_results = results_df.merge(labels_df, on='filename', how='inner')
    
    if len(public_results) > 0:
        public_accuracy = (public_results['predicted_is_illegal'] == public_results['is_illegal']).mean()
        print(f"Accuracy on public test data: {public_accuracy:.4f}")
        print(f"Evaluated on {len(public_results)} public test sessions")
        
        # Detailed evaluation
        print("\nDetailed Classification Report:")
        print(training_results['classification_report'])
        
        print("\nConfusion Matrix:")
        print(training_results['confusion_matrix'])
    else:
        print("No matching sessions found between predictions and public labels")
    
    # 8. Save results
    print("\nStep 7: Saving results...")
    results_df.to_csv('predictions.csv', index=False)
    print("Predictions saved to 'predictions.csv'")
    
    if len(public_results) > 0:
        public_results.to_csv('public_evaluation.csv', index=False)
        print("Public evaluation saved to 'public_evaluation.csv'")
    
    # 9. Save trained model
    classifier.save_model('mouse_dynamics_model.pkl')
    
    # 10. Summary statistics
    print("\n=== SUMMARY ===")
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
    print("\nTest data:")
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
    print(f"\nPublic labels: {len(labels_df)} sessions")
    print(f"Legal: {(labels_df['is_illegal'] == 0).sum()}")
    print(f"Illegal: {(labels_df['is_illegal'] == 1).sum()}")


if __name__ == "__main__":
    # Run data analysis first
    analyze_data_distribution()
    
    print("\n" + "="*50 + "\n")
    
    # Run main classification pipeline
    classifier, results, training_results = main()
    
    print("\nPipeline completed successfully!")
    print("Check 'predictions.csv' for test predictions")
    print("Check 'mouse_dynamics_model.pkl' for the saved model")
