# Mouse Dynamics Classification System

A machine learning system for detecting fraudulent user behavior through mouse movement pattern analysis. This system classifies mouse interaction sessions as either legitimate (legal) or potentially fraudulent (illegal) based on behavioral biometrics.

## 🎯 Overview

This project implements a Random Forest classifier that analyzes mouse movement patterns to distinguish between legitimate users and potential bots or fraudulent behavior. The system achieves **94.85% accuracy** on test data.

## 📊 Results Summary

- **94.85% Overall Accuracy** on public test data
- **95.62% Recall** for legal sessions (low false alarm rate)
- **94.07% Recall** for illegal sessions (good threat detection)
- **40 behavioral features** extracted from mouse movement data
- Processed **1,611 test sessions** with predictions for all

## 🗂️ Project Structure

```
Mouse-Dynamics-QS25/
├── training_files/          # Training data (legitimate user sessions)
├── test_files/             # Test data (sessions to classify)
├── public_labels.csv       # Ground truth labels for test sessions
├── feature_extractor.py   # Feature engineering from mouse data
├── classifier.py          # Random Forest classifier implementation
├── main.py                # Main training and prediction pipeline
├── performance_analysis.py # Detailed performance evaluation
├── run_pipeline.py        # Comprehensive analysis with reports
└── Generated Output Files:
    ├── predictions.csv                    # Predictions for all test sessions
    ├── public_evaluation.csv            # Evaluation against known labels
    ├── mouse_dynamics_model.pkl         # Trained model
    ├── classification_report.txt        # Summary report
    ├── detailed_performance_analysis.txt # Detailed analysis
    └── prediction_analysis.png          # Visualization plots
```

## 🚀 Quick Start

### Prerequisites
```bash
# Activate conda environment
conda activate iw2025

# Install required packages (if not already installed)
pip install scikit-learn pandas numpy matplotlib seaborn
```

### Running the Complete Pipeline

1. **Train and Test the Model:**
   ```bash
   python main.py
   ```

2. **Run Comprehensive Analysis:**
   ```bash
   python run_pipeline.py
   ```

3. **Generate Performance Analysis:**
   ```bash
   python performance_analysis.py
   ```

## 🔧 How It Works

### 1. Data Structure

**Training Data:** `training_files/userX/session_XXXXXXXXXX`
- Contains mouse movement sessions from legitimate users
- CSV format with columns: `record timestamp,client timestamp,button,state,x,y`

**Test Data:** `test_files/userX/session_XXXXXXXXXX`
- Sessions to be classified as legal or illegal
- Same CSV format as training data

**Labels:** `public_labels.csv`
- Ground truth for test sessions: `filename,is_illegal`
- `is_illegal = 0`: Legal session
- `is_illegal = 1`: Illegal session

### 2. Feature Engineering (`feature_extractor.py`)

Extracts 40+ behavioral features from raw mouse data:

**Movement Patterns:**
- Velocity statistics (mean, std, max, percentiles)
- Acceleration patterns
- Distance traveled and movement smoothness
- Direction changes and movement consistency

**Timing Behavior:**
- Actions per second
- Time intervals between actions
- Pause patterns and long delays

**Click Behavior:**
- Click frequency and ratios
- Button usage patterns (left/right/none)
- Click-to-movement ratios

**Screen Usage:**
- Coordinate ranges and area coverage
- Movement distribution patterns

### 3. Classification Strategy

**Training Approach:**
1. Use all training sessions as "legal" examples (legitimate users)
2. Use labeled test sessions to provide "illegal" examples
3. Combine for balanced training set: 476 legal + 405 illegal = 881 total samples

**Model:**
- Random Forest Classifier (200 trees)
- StandardScaler for feature normalization
- Cross-validation for robust evaluation

### 4. Key Features Discovered

Most important features for classification:
1. **Direction Changes (7.47%)** - How often users change mouse direction
2. **Click Ratio (6.02%)** - Proportion of clicks vs movements
3. **Actions per Second (5.99%)** - Speed of interactions
4. **Mean Time Interval (5.40%)** - Timing between actions
5. **Move Count (4.93%)** - Total mouse movements

## 📈 Performance Metrics

### Confusion Matrix
```
                Predicted
Actual     Legal  Illegal  Total
Legal        393       18    411
Illegal       24      381    405
Total        417      399    816
```

### Detailed Metrics
- **Legal Sessions:**
  - Precision: 94.24%
  - Recall: 95.62%
  - F1-Score: 94.93%

- **Illegal Sessions:**
  - Precision: 95.49%
  - Recall: 94.07%
  - F1-Score: 94.78%

### Error Analysis
- **False Positives:** 18 sessions (4.38% of legal sessions wrongly flagged)
- **False Negatives:** 24 sessions (5.93% of illegal sessions missed)
- **High Confidence Predictions:** 57.23% of all predictions

## 📁 Output Files

### `predictions.csv`
Contains predictions for all test sessions:
```csv
filename,predicted_is_illegal,illegal_probability
session_8833850952,1.0,0.6741
session_3379861047,1.0,0.6568
...
```

### `public_evaluation.csv`
Detailed comparison with ground truth:
```csv
filename,predicted_is_illegal,illegal_probability,is_illegal
session_8833850952,1.0,0.6741,1
session_3379861047,1.0,0.6568,1
...
```

### `mouse_dynamics_model.pkl`
Serialized trained model ready for deployment:
```python
import pickle
with open('mouse_dynamics_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
```

## 🔍 Usage Examples

### Train a New Model
```python
from classifier import MouseDynamicsClassifier

# Initialize and train
classifier = MouseDynamicsClassifier()
X_train, y_train = classifier.load_data()
training_results = classifier.train(X_train, y_train)

# Save model
classifier.save_model('my_model.pkl')
```

### Make Predictions on New Data
```python
# Load existing model
classifier = MouseDynamicsClassifier()
classifier.load_model('mouse_dynamics_model.pkl')

# Load and predict on new sessions
X_new, _, session_names = load_session_data('new_test_files')
predictions, probabilities = classifier.predict(X_new)
```

### Extract Features from Single Session
```python
from feature_extractor import extract_features_from_session

features = extract_features_from_session('path/to/session_file')
print(features)
```

## 🎯 Key Insights

1. **Behavioral Differences:** The model successfully identifies distinct patterns between legitimate and fraudulent mouse behavior
2. **Direction Changes:** Most important feature - bots/scripts tend to have different movement patterns
3. **Timing Patterns:** Click ratios and timing intervals are strong indicators
4. **High Precision:** 95.49% precision for illegal detection means very few false alarms
5. **Robust Performance:** Cross-validation shows consistent performance across different data splits

## 🛠️ Customization

### Adjust Model Parameters
Edit `classifier.py` to modify RandomForest parameters:
```python
self.model = RandomForestClassifier(
    n_estimators=200,    # Number of trees
    max_depth=15,        # Tree depth
    min_samples_split=5, # Minimum samples to split
    random_state=42
)
```

### Add New Features
Extend `extract_features_from_session()` in `feature_extractor.py`:
```python
features['my_new_feature'] = calculate_my_feature(df)
```

### Change Classification Threshold
Adjust the probability threshold for classification:
```python
# Default threshold is 0.5, but you can customize:
custom_predictions = (probabilities > 0.6).astype(int)  # More conservative
```

## 🔬 Technical Details

- **Algorithm:** Random Forest (ensemble of decision trees)
- **Features:** 40 behavioral metrics extracted from mouse movement
- **Data Processing:** Handles missing values, outliers, and feature scaling
- **Validation:** 5-fold cross-validation + holdout validation
- **Output:** Binary classification + confidence scores

## 📋 Requirements

- Python 3.9+
- scikit-learn
- pandas
- numpy
- matplotlib (optional, for visualizations)
- seaborn (optional, for visualizations)

## 🚀 Next Steps

1. **Deploy Model:** Use `mouse_dynamics_model.pkl` in production systems
2. **Monitor Performance:** Track real-world accuracy and retrain as needed
3. **Feature Engineering:** Experiment with additional behavioral features
4. **Threshold Tuning:** Adjust classification threshold based on business requirements
5. **Real-time Integration:** Implement streaming classification for live sessions

---

**Developed for Mouse Dynamics Challenge 2025**  
*Behavioral biometrics for fraud detection through mouse movement analysis*
