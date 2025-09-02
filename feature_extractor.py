"""
Feature extraction module for mouse dynamics data
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, Optional, List, Tuple


def extract_features_from_session(file_path: str) -> Optional[Dict]:
    """
    Extract behavioral features from a single mouse session file
    
    Args:
        file_path: Path to the session CSV file
        
    Returns:
        Dictionary of extracted features or None if processing fails
    """
    try:
        df = pd.read_csv(file_path)
        
        # Basic validation
        if len(df) < 2:
            return None
            
        # Basic session statistics
        total_actions = len(df)
        total_time = df['record timestamp'].max() - df['record timestamp'].min()
        
        # Calculate movement deltas
        df['dx'] = df['x'].diff()
        df['dy'] = df['y'].diff()
        df['distance'] = np.sqrt(df['dx']**2 + df['dy']**2)
        df['dt'] = df['record timestamp'].diff()
        
        # Calculate velocity and acceleration
        df['velocity'] = df['distance'] / df['dt'].replace(0, np.nan)
        df['acceleration'] = df['velocity'].diff() / df['dt'].replace(0, np.nan)
        
        # Clean up infinite and NaN values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Skip if no valid data remains
        velocity_clean = df['velocity'].dropna()
        acceleration_clean = df['acceleration'].dropna()
        distance_clean = df['distance'].dropna()
        dt_clean = df['dt'].dropna()
        
        if len(velocity_clean) == 0 or len(distance_clean) == 0:
            return None
        
        # Extract features
        features = {
            # Basic session characteristics
            'total_actions': total_actions,
            'session_duration': max(total_time, 0.001),
            'actions_per_second': total_actions / max(total_time, 0.001),
            
            # Movement distance features
            'mean_distance': distance_clean.mean(),
            'std_distance': distance_clean.std(),
            'max_distance': distance_clean.max(),
            'min_distance': distance_clean.min(),
            'total_distance': distance_clean.sum(),
            'distance_percentile_75': distance_clean.quantile(0.75),
            'distance_percentile_25': distance_clean.quantile(0.25),
            
            # Velocity features
            'mean_velocity': velocity_clean.mean(),
            'std_velocity': velocity_clean.std(),
            'max_velocity': velocity_clean.max(),
            'min_velocity': velocity_clean.min(),
            'velocity_percentile_75': velocity_clean.quantile(0.75),
            'velocity_percentile_25': velocity_clean.quantile(0.25),
            
            # Acceleration features
            'mean_acceleration': acceleration_clean.mean() if len(acceleration_clean) > 0 else 0,
            'std_acceleration': acceleration_clean.std() if len(acceleration_clean) > 0 else 0,
            'max_acceleration': acceleration_clean.max() if len(acceleration_clean) > 0 else 0,
            
            # Screen area usage
            'x_range': df['x'].max() - df['x'].min(),
            'y_range': df['y'].max() - df['y'].min(),
            'x_mean': df['x'].mean(),
            'y_mean': df['y'].mean(),
            'x_std': df['x'].std(),
            'y_std': df['y'].std(),
            
            # Click behavior analysis
            'click_count': len(df[df['state'] == 'Pressed']),
            'move_count': len(df[df['state'] == 'Move']),
            'release_count': len(df[df['state'] == 'Released']),
            'drag_count': len(df[df['state'] == 'Drag']) if 'Drag' in df['state'].values else 0,
            
            # Click ratios
            'click_ratio': len(df[df['state'] == 'Pressed']) / total_actions,
            'move_ratio': len(df[df['state'] == 'Move']) / total_actions,
            
            # Timing patterns
            'mean_time_interval': dt_clean.mean() if len(dt_clean) > 0 else 0,
            'std_time_interval': dt_clean.std() if len(dt_clean) > 0 else 0,
            'max_time_interval': dt_clean.max() if len(dt_clean) > 0 else 0,
            
            # Pause behavior (intervals longer than 95th percentile)
            'long_pauses': len(dt_clean[dt_clean > dt_clean.quantile(0.95)]) if len(dt_clean) > 0 else 0,
            'pause_ratio': len(dt_clean[dt_clean > dt_clean.quantile(0.95)]) / len(dt_clean) if len(dt_clean) > 0 else 0,
            
            # Movement smoothness (direction changes)
            'direction_changes': calculate_direction_changes(df),
            
            # Button usage patterns
            'left_button_ratio': len(df[df['button'] == 'Button1']) / total_actions if 'Button1' in df['button'].values else 0,
            'right_button_ratio': len(df[df['button'] == 'Button2']) / total_actions if 'Button2' in df['button'].values else 0,
            'no_button_ratio': len(df[df['button'] == 'NoButton']) / total_actions,
        }
        
        return features
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def calculate_direction_changes(df: pd.DataFrame) -> float:
    """
    Calculate the number of significant direction changes in mouse movement
    """
    try:
        # Calculate angles between consecutive movements
        dx = df['dx'].fillna(0)
        dy = df['dy'].fillna(0)
        
        # Calculate angle changes
        angles = np.arctan2(dy, dx)
        angle_changes = np.abs(np.diff(angles))
        
        # Count significant direction changes (> 45 degrees)
        significant_changes = np.sum(angle_changes > np.pi/4)
        
        return significant_changes / max(len(df) - 1, 1)
        
    except:
        return 0


def load_session_data(data_dir: str, labels_dict: Dict[str, int] = None) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Load and extract features from all sessions in a directory
    
    Args:
        data_dir: Directory containing user subdirectories with session files
        labels_dict: Dictionary mapping session names to labels (for training data)
        
    Returns:
        Tuple of (features_df, labels_array, session_names)
    """
    all_features = []
    all_labels = []
    session_names = []
    
    # Get all session files
    session_files = []
    for user_dir in os.listdir(data_dir):
        user_path = os.path.join(data_dir, user_dir)
        if os.path.isdir(user_path):
            for session_file in os.listdir(user_path):
                session_files.append(os.path.join(user_path, session_file))
    
    print(f"Found {len(session_files)} sessions in {data_dir}")
    
    processed_count = 0
    for i, file_path in enumerate(session_files):
        session_name = os.path.basename(file_path)
        
        # For training data, only process sessions with labels
        if labels_dict is not None and session_name not in labels_dict:
            continue
            
        features = extract_features_from_session(file_path)
        if features is not None:
            all_features.append(features)
            session_names.append(session_name)
            
            if labels_dict is not None:
                all_labels.append(labels_dict[session_name])
            
            processed_count += 1
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(session_files)} files, extracted {processed_count} valid sessions")
    
    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)
    labels_array = np.array(all_labels) if all_labels else np.array([])
    
    print(f"Successfully extracted features from {len(features_df)} sessions")
    if len(labels_array) > 0:
        print(f"Legal sessions: {(labels_array == 0).sum()}")
        print(f"Illegal sessions: {(labels_array == 1).sum()}")
    
    return features_df, labels_array, session_names
