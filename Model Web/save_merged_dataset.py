#!/usr/bin/env python3
"""
Save the merged dataset for inspection and analysis.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

def read_csv_with_comments(filepath: str) -> pd.DataFrame:
    """Read CSV file, skipping comment lines starting with '#'."""
    return pd.read_csv(filepath, comment='#')


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


def main():
    """Create and save the merged dataset."""
    print("📊 Creating merged dataset...")
    
    # File paths
    kepler_path = "dataset/Kepler Objects of Interest (KOI)_2025.09.29_04.48.01.csv"
    k2_path = "dataset/K2 Planets and Candidates_2025.09.29_04.49.17.csv"
    tess_path = "dataset/TESS Objects of Interest (TOI)_2025.09.29_04.48.50.csv"
    
    # Read datasets
    print("Reading individual datasets...")
    kepler_df = read_csv_with_comments(kepler_path)
    k2_df = read_csv_with_comments(k2_path)
    tess_df = read_csv_with_comments(tess_path)
    
    print(f"Kepler: {len(kepler_df)} rows")
    print(f"K2: {len(k2_df)} rows")
    print(f"TESS: {len(tess_df)} rows")
    
    # Process each dataset
    print("Processing datasets...")
    kepler_processed = process_kepler(kepler_df)
    k2_processed = process_k2(k2_df)
    tess_processed = process_tess(tess_df)
    
    # Combine all datasets
    unified_df = pd.concat([kepler_processed, k2_processed, tess_processed], 
                          ignore_index=True)
    
    print(f"Unified dataset: {len(unified_df)} rows")
    
    # Create features
    df_with_features = create_features(unified_df)
    
    # Create output directory
    os.makedirs("data/processed", exist_ok=True)
    
    # Clean data types for saving
    df_clean = df_with_features.copy()
    
    # Convert object_id to string to avoid parquet issues
    df_clean['object_id'] = df_clean['object_id'].astype(str)
    
    # Save the merged dataset
    output_path = "data/processed/merged_exoplanet_dataset.csv"
    df_clean.to_csv(output_path, index=False)
    
    print(f"✅ Merged dataset saved to:")
    print(f"   📄 CSV: {output_path}")
    
    # Show dataset info
    print(f"\n📊 Dataset Information:")
    print(f"   Total rows: {len(df_with_features):,}")
    print(f"   Total columns: {len(df_with_features.columns)}")
    print(f"   Columns: {list(df_with_features.columns)}")
    
    print(f"\n🏷️  Label Distribution:")
    print(df_with_features['label'].value_counts())
    
    print(f"\n🛰️  Mission Distribution:")
    print(df_with_features['mission'].value_counts())
    
    # Show sample data
    print(f"\n📋 Sample Data (first 5 rows):")
    print(df_with_features.head())
    
    # Show data types
    print(f"\n🔢 Data Types:")
    print(df_with_features.dtypes)
    
    return df_with_features


if __name__ == "__main__":
    merged_df = main()
