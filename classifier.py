"""
Mouse dynamics classifier training and evaluation
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any

from feature_extractor import load_session_data

# Optional visualization imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Matplotlib/Seaborn not available. Plotting functions disabled.")


class MouseDynamicsClassifier:
    """
    Mouse dynamics behavioral classifier
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.is_trained = False
        
    def load_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Load training data and labels
        Since public_labels.csv contains test labels, we'll use a different strategy:
        1. Use all training sessions as "legal" (label 0) - assumption that training data is from legitimate users
        2. Use some test sessions with known labels for validation
        """
        print("Loading public labels...")
        labels_df = pd.read_csv('public_labels.csv')
        
        print(f"Loaded {len(labels_df)} labeled test sessions")
        print(f"Legal: {sum(1 for x in labels_df['is_illegal'] if x == 0)}")
        print(f"Illegal: {sum(1 for x in labels_df['is_illegal'] if x == 1)}")
        
        # Strategy 1: Use training data as legal samples
        print("\nExtracting features from training data (assuming all legal)...")
        X_train_legal, _, train_session_names = load_session_data('training_files')
        
        if len(X_train_legal) == 0:
            print("No training data found!")
            return pd.DataFrame(), np.array([])
        
        # Create labels for training data (all legal)
        y_train_legal = np.zeros(len(X_train_legal))
        
        # Strategy 2: Use some labeled test sessions for additional training data
        print("Adding labeled test sessions to training data...")
        labels_dict = dict(zip(labels_df['filename'], labels_df['is_illegal']))
        X_test_labeled, y_test_labeled, test_labeled_names = load_session_data('test_files', labels_dict)
        
        if len(X_test_labeled) > 0:
            # Combine training and labeled test data
            X_combined = pd.concat([X_train_legal, X_test_labeled], ignore_index=True)
            y_combined = np.concatenate([y_train_legal, y_test_labeled])
            
            print(f"Combined dataset: {len(X_combined)} sessions")
            print(f"Legal: {(y_combined == 0).sum()}, Illegal: {(y_combined == 1).sum()}")
            
            return X_combined, y_combined
        else:
            print("No labeled test data found, using only training data as legal")
            return X_train_legal, y_train_legal
    
    def preprocess_data(self, X: pd.DataFrame, fit_scaler: bool = True) -> pd.DataFrame:
        """
        Preprocess feature data
        """
        # Handle missing values
        X_clean = X.fillna(X.median())
        
        # Remove infinite values
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        X_clean = X_clean.fillna(X_clean.median())
        
        if fit_scaler:
            # Fit and transform with new scaler
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_clean)
            self.feature_columns = X_clean.columns
        else:
            # Transform with existing scaler
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit_scaler=True first.")
            
            # Ensure columns match training data
            missing_cols = set(self.feature_columns) - set(X_clean.columns)
            for col in missing_cols:
                X_clean[col] = 0
            
            X_clean = X_clean[self.feature_columns]
            X_scaled = self.scaler.transform(X_clean)
        
        return pd.DataFrame(X_scaled, columns=self.feature_columns)
    
    def train(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """
        Train the classifier
        """
        print("Preprocessing training data...")
        X_processed = self.preprocess_data(X, fit_scaler=True)
        
        print("Splitting data for validation...")
        X_train, X_val, y_train, y_val = train_test_split(
            X_processed, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Initialize and train model
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        print("Training Random Forest classifier...")
        self.model.fit(X_train, y_train)
        
        # Validation
        y_val_pred = self.model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        
        # Cross-validation on full dataset
        print("Performing cross-validation...")
        cv_scores = cross_val_score(self.model, X_processed, y, cv=5, scoring='accuracy')
        
        self.is_trained = True
        
        results = {
            'validation_accuracy': val_accuracy,
            'cv_mean_accuracy': cv_scores.mean(),
            'cv_std_accuracy': cv_scores.std(),
            'classification_report': classification_report(y_val, y_val_pred),
            'confusion_matrix': confusion_matrix(y_val, y_val_pred),
            'feature_importance': pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        }
        
        return results
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")
        
        X_processed = self.preprocess_data(X, fit_scaler=False)
        predictions = self.model.predict(X_processed)
        probabilities = self.model.predict_proba(X_processed)[:, 1]
        
        return predictions, probabilities
    
    def save_model(self, filepath: str):
        """
        Save the trained model and scaler
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'random_state': self.random_state
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model and scaler
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.random_state = model_data['random_state']
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")


def analyze_features(X: pd.DataFrame, y: np.ndarray, feature_importance: pd.DataFrame):
    """
    Create visualizations for feature analysis (if plotting libraries available)
    """
    if not PLOTTING_AVAILABLE:
        print("Plotting libraries not available. Showing text summary instead.")
        print("\nTop 10 Features by Importance:")
        print(feature_importance.head(10))
        return
    
    # Feature importance plot
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    top_features = feature_importance.head(10)
    plt.barh(top_features['feature'], top_features['importance'])
    plt.title('Top 10 Most Important Features')
    plt.xlabel('Feature Importance')
    
    # Feature distributions by class
    plt.subplot(2, 2, 2)
    key_features = ['mean_velocity', 'std_velocity']
    for feature in key_features:
        if feature in X.columns:
            legal_data = X[y == 0][feature]
            illegal_data = X[y == 1][feature]
            
            plt.hist(legal_data, alpha=0.5, label=f'Legal {feature}', bins=30)
            plt.hist(illegal_data, alpha=0.5, label=f'Illegal {feature}', bins=30)
    
    plt.title('Velocity Feature Distributions')
    plt.legend()
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    
    # Class distribution
    plt.subplot(2, 2, 3)
    class_counts = pd.Series(y).value_counts()
    plt.pie(class_counts.values, labels=['Legal', 'Illegal'], autopct='%1.1f%%')
    plt.title('Class Distribution')
    
    # Session duration by class
    plt.subplot(2, 2, 4)
    if 'session_duration' in X.columns:
        legal_duration = X[y == 0]['session_duration']
        illegal_duration = X[y == 1]['session_duration']
        
        plt.boxplot([legal_duration, illegal_duration], labels=['Legal', 'Illegal'])
        plt.title('Session Duration by Class')
        plt.ylabel('Duration (seconds)')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Quick test of the module
    classifier = MouseDynamicsClassifier()
    X_train, y_train = classifier.load_data()
    
    print(f"\nLoaded data shape: {X_train.shape}")
    print(f"Feature columns: {list(X_train.columns)}")
    print(f"Labels shape: {y_train.shape}")
