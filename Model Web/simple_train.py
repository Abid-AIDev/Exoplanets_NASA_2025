#!/usr/bin/env python3
"""
Simplified training pipeline for exoplanet detection models.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Model imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
)
import joblib

# XGBoost and LightGBM
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


def read_csv_with_comments(filepath: str) -> pd.DataFrame:
    """Read CSV file, skipping comment lines starting with '#'."""
    return pd.read_csv(filepath, comment='#')


def process_datasets():
    """Process and merge the three exoplanet datasets."""
    print("📊 Processing and merging datasets...")
    
    # File paths
    kepler_path = "dataset/Kepler Objects of Interest (KOI)_2025.09.29_04.48.01.csv"
    k2_path = "dataset/K2 Planets and Candidates_2025.09.29_04.49.17.csv"
    tess_path = "dataset/TESS Objects of Interest (TOI)_2025.09.29_04.48.50.csv"
    
    # Read datasets
    kepler_df = read_csv_with_comments(kepler_path)
    k2_df = read_csv_with_comments(k2_path)
    tess_df = read_csv_with_comments(tess_path)
    
    print(f"Kepler: {len(kepler_df)} rows")
    print(f"K2: {len(k2_df)} rows")
    print(f"TESS: {len(tess_df)} rows")
    
    # Process Kepler
    kepler_processed = process_kepler(kepler_df)
    
    # Process K2
    k2_processed = process_k2(k2_df)
    
    # Process TESS
    tess_processed = process_tess(tess_df)
    
    # Combine all datasets
    unified_df = pd.concat([kepler_processed, k2_processed, tess_processed], 
                          ignore_index=True)
    
    print(f"Unified dataset: {len(unified_df)} rows")
    
    return unified_df


def process_kepler(df):
    """Process Kepler dataset."""
    processed = df.copy()
    
    # Rename columns
    column_mapping = {
        'kepoi_name': 'object_id',
        'koi_period': 'period_days',
        'koi_duration': 'duration_hours', 
        'koi_depth': 'depth_ppm',
        'koi_prad': 'planet_radius_rearth',
        'koi_score': 'score',
        'koi_disposition': 'label_raw',
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in processed.columns:
            processed[new_col] = processed[old_col]
    
    processed['mission'] = 'kepler'
    
    # Map disposition
    disposition_map = {
        'CONFIRMED': 'planet',
        'CANDIDATE': 'candidate', 
        'FALSE POSITIVE': 'fp'
    }
    processed['label'] = processed['label_raw'].map(disposition_map).fillna('unknown')
    
    # Select columns
    columns_to_keep = [
        'object_id', 'mission', 'period_days', 'duration_hours', 'depth_ppm',
        'planet_radius_rearth', 'score', 'label', 'label_raw'
    ]
    
    return processed[[col for col in columns_to_keep if col in processed.columns]]


def process_k2(df):
    """Process K2 dataset."""
    processed = df.copy()
    
    # Rename columns
    column_mapping = {
        'epic_name': 'object_id',
        'pl_orbper': 'period_days',
        'pl_trandurh': 'duration_hours',
        'pl_trandep': 'depth_ppm', 
        'pl_rade': 'planet_radius_rearth',
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in processed.columns:
            processed[new_col] = processed[old_col]
    
    processed['mission'] = 'k2'
    
    # Find disposition column
    disposition_col = None
    for col in ['disposition', 'k2_disposition', 'archive_disposition']:
        if col in processed.columns:
            disposition_col = col
            break
    
    if disposition_col:
        processed['label_raw'] = processed[disposition_col]
        disposition_map = {
            'CONFIRMED': 'planet',
            'CANDIDATE': 'candidate',
            'FALSE POSITIVE': 'fp',
            'FP': 'fp'
        }
        processed['label'] = processed['label_raw'].map(disposition_map).fillna('unknown')
    else:
        processed['label'] = 'unknown'
        processed['label_raw'] = 'unknown'
    
    # Select columns
    columns_to_keep = [
        'object_id', 'mission', 'period_days', 'duration_hours', 'depth_ppm',
        'planet_radius_rearth', 'label', 'label_raw'
    ]
    
    return processed[[col for col in columns_to_keep if col in processed.columns]]


def process_tess(df):
    """Process TESS dataset."""
    processed = df.copy()
    
    # Rename columns
    column_mapping = {
        'toi': 'object_id',
        'pl_orbper': 'period_days',
        'pl_trandurh': 'duration_hours',
        'pl_trandep': 'depth_ppm',
        'pl_rade': 'planet_radius_rearth',
        'tfopwg_disp': 'label_raw'
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in processed.columns:
            processed[new_col] = processed[old_col]
    
    processed['mission'] = 'tess'
    
    # Map disposition
    disposition_map = {
        'CP': 'planet',
        'PC': 'candidate',
        'FP': 'fp',
        'KP': 'planet',
        'APC': 'candidate'
    }
    processed['label'] = processed['label_raw'].map(disposition_map).fillna('unknown')
    
    # Select columns
    columns_to_keep = [
        'object_id', 'mission', 'period_days', 'duration_hours', 'depth_ppm',
        'planet_radius_rearth', 'label', 'label_raw'
    ]
    
    return processed[[col for col in columns_to_keep if col in processed.columns]]


def create_features(df):
    """Create additional features."""
    df = df.copy()
    
    # Convert to numeric
    numeric_cols = ['period_days', 'duration_hours', 'depth_ppm', 'planet_radius_rearth', 'score']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Create derived features
    if 'period_days' in df.columns and 'duration_hours' in df.columns:
        df['duty_cycle'] = df['duration_hours'] / (df['period_days'] * 24)
        df['transit_frequency'] = 1.0 / df['period_days']
    
    if 'depth_ppm' in df.columns:
        df['planet_radius_from_depth'] = np.sqrt(df['depth_ppm'] / 1e6)
    
    # Mission encoding
    mission_encoding = {'kepler': 0, 'k2': 1, 'tess': 2}
    df['mission_encoded'] = df['mission'].map(mission_encoding).fillna(-1)
    
    return df


def train_models(X_train, X_test, y_train, y_test, feature_columns, label_encoder):
    """Train multiple models and select the best one."""
    print("🤖 Training models...")
    
    # Define models
    models = {}
    
    # Random Forest
    models['RandomForest'] = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # Gradient Boosting
    models['GradientBoosting'] = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    
    # Logistic Regression with scaling
    models['LogisticRegression'] = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            random_state=42,
            max_iter=1000,
            multi_class='ovr'
        ))
    ])
    
    # SVM with scaling
    models['SVM'] = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(
            random_state=42,
            probability=True,
            kernel='rbf'
        ))
    ])
    
    # XGBoost if available
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss'
        )
    
    # LightGBM if available
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )
    
    # Train and evaluate models
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        try:
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            
            # ROC-AUC
            try:
                if len(np.unique(y_test)) > 2:
                    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
                else:
                    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            except:
                roc_auc = 0.0
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc
            }
            
            trained_models[name] = model
            
            print(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            
        except Exception as e:
            print(f"Error training {name}: {e}")
            continue
    
    return results, trained_models


def main():
    """Main training pipeline."""
    print("🚀 Starting Exoplanet Detection Model Training")
    print("=" * 50)
    
    # Create output directories
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Step 1: Process datasets
    unified_df = process_datasets()
    
    # Step 2: Create features
    df_with_features = create_features(unified_df)
    
    # Step 3: Prepare training data
    print("📊 Preparing training data...")
    
    # Remove rows with missing labels
    df_clean = df_with_features.dropna(subset=['label']).copy()
    df_clean = df_clean[df_clean['label'] != 'unknown'].copy()
    
    print(f"Clean dataset: {len(df_clean)} rows")
    print(f"Label distribution:\n{df_clean['label'].value_counts()}")
    
    # Define feature columns
    feature_columns = [
        'period_days', 'duration_hours', 'depth_ppm', 'planet_radius_rearth',
        'duty_cycle', 'transit_frequency', 'planet_radius_from_depth', 'mission_encoded'
    ]
    
    # Add score if available
    if 'score' in df_clean.columns:
        feature_columns.append('score')
    
    # Select available features
    available_features = [col for col in feature_columns if col in df_clean.columns]
    print(f"Using features: {available_features}")
    
    # Prepare features and target
    X = df_clean[available_features].copy()
    y = df_clean['label'].copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Create train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Step 4: Train models
    results, trained_models = train_models(
        X_train, X_test, y_train, y_test, available_features, label_encoder
    )
    
    # Step 5: Select best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['f1'])
    best_model = trained_models[best_model_name]
    best_score = results[best_model_name]['f1']
    
    print(f"\n🏆 Best model: {best_model_name} (F1: {best_score:.4f})")
    
    # Step 6: Generate detailed report
    y_pred = best_model.predict(X_test)
    print(f"\n📋 Detailed Report for {best_model_name}:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Step 7: Save best model
    model_data = {
        'model': best_model,
        'feature_columns': available_features,
        'label_encoder': label_encoder,
        'model_name': best_model_name,
        'metrics': results[best_model_name],
        'training_data_info': {
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'n_features': len(available_features),
            'n_classes': len(label_encoder.classes_)
        }
    }
    
    joblib.dump(model_data, "models/best_model.joblib")
    print(f"✅ Best model saved to models/best_model.joblib")
    
    # Save results
    with open("data/processed/training_results.json", 'w') as f:
        json.dump({
            'model_results': {k: {kk: float(vv) if isinstance(vv, (np.integer, np.floating)) else vv 
                                for kk, vv in v.items()} for k, v in results.items()},
            'best_model': best_model_name,
            'best_score': float(best_score),
            'feature_columns': available_features,
            'label_classes': label_encoder.classes_.tolist()
        }, f, indent=2)
    
    # Save training summary
    summary = {
        "training_summary": {
            "best_model": best_model_name,
            "best_f1_score": float(best_score),
            "total_models_trained": len(trained_models),
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "features_used": available_features,
            "classes": label_encoder.classes_.tolist()
        },
        "model_performance": {k: {kk: float(vv) if isinstance(vv, (np.integer, np.floating)) else vv 
                                for kk, vv in v.items()} for k, v in results.items()}
    }
    
    with open("data/processed/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print final summary
    print("\n" + "="*50)
    print("📊 TRAINING SUMMARY")
    print("="*50)
    print(f"🏆 Best Model: {best_model_name}")
    print(f"📈 Best F1-Score: {best_score:.4f}")
    print(f"🔢 Total Models Trained: {len(trained_models)}")
    print(f"📊 Training Samples: {len(X_train):,}")
    print(f"🧪 Test Samples: {len(X_test):,}")
    print(f"🎯 Features Used: {len(available_features)}")
    print(f"🏷️  Classes: {', '.join(label_encoder.classes_)}")
    
    print("\n📋 Model Performance Comparison:")
    print("-" * 50)
    print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 50)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<20} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
              f"{metrics['recall']:<10.4f} {metrics['f1']:<10.4f}")
    
    print("\n✅ Model saved to: models/best_model.joblib")
    print("✅ Results saved to: data/processed/")
    print("✅ Ready for Streamlit app!")
    
    return best_model_name, best_score


if __name__ == "__main__":
    main()
