#!/usr/bin/env python3
"""
Complete pipeline for training exoplanet detection models.
This script processes the datasets, trains multiple models, and saves the best one.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data.process_datasets import main as process_datasets
from models.train_models import main as train_models


def main():
    """Complete training pipeline."""
    print("🚀 Starting Exoplanet Detection Model Training Pipeline")
    print("=" * 60)
    
    # Step 1: Process datasets
    print("\n📊 Step 1: Processing and merging datasets...")
    try:
        X_train, X_test, y_train, y_test, feature_columns, label_encoder = process_datasets()
        print("✅ Dataset processing completed successfully!")
    except Exception as e:
        print(f"❌ Error in dataset processing: {e}")
        return
    
    # Step 2: Train models
    print("\n🤖 Step 2: Training and evaluating models...")
    try:
        trainer, best_model_name = train_models()
        print("✅ Model training completed successfully!")
    except Exception as e:
        print(f"❌ Error in model training: {e}")
        return
    
    # Step 3: Generate final summary
    print("\n📋 Step 3: Generating final summary...")
    generate_final_summary(trainer, best_model_name)
    
    print("\n🎉 Pipeline completed successfully!")
    print("Your trained model is ready for the Streamlit app!")


def generate_final_summary(trainer, best_model_name):
    """Generate a comprehensive summary of the training results."""
    
    # Create summary report
    summary = {
        "training_summary": {
            "best_model": best_model_name,
            "best_f1_score": float(trainer.best_score),
            "total_models_trained": len(trainer.models),
            "training_samples": len(trainer.X_train),
            "test_samples": len(trainer.X_test),
            "features_used": trainer.feature_columns,
            "classes": trainer.label_encoder.classes_.tolist()
        },
        "model_performance": {}
    }
    
    # Add performance metrics for all models
    for model_name, metrics in trainer.results.items():
        summary["model_performance"][model_name] = {
            "accuracy": float(metrics["accuracy"]),
            "precision_macro": float(metrics["precision_macro"]),
            "recall_macro": float(metrics["recall_macro"]),
            "f1_macro": float(metrics["f1_macro"]),
            "roc_auc_ovr": float(metrics.get("roc_auc_ovr", 0.0))
        }
    
    # Save summary
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print summary to console
    print("\n" + "="*60)
    print("📊 TRAINING SUMMARY")
    print("="*60)
    print(f"🏆 Best Model: {best_model_name}")
    print(f"📈 Best F1-Score: {trainer.best_score:.4f}")
    print(f"🔢 Total Models Trained: {len(trainer.models)}")
    print(f"📊 Training Samples: {len(trainer.X_train):,}")
    print(f"🧪 Test Samples: {len(trainer.X_test):,}")
    print(f"🎯 Features Used: {len(trainer.feature_columns)}")
    print(f"🏷️  Classes: {', '.join(trainer.label_encoder.classes_)}")
    
    print("\n📋 Model Performance Comparison:")
    print("-" * 60)
    print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 60)
    
    for model_name, metrics in trainer.results.items():
        print(f"{model_name:<20} {metrics['accuracy']:<10.4f} {metrics['precision_macro']:<10.4f} "
              f"{metrics['recall_macro']:<10.4f} {metrics['f1_macro']:<10.4f}")
    
    print("\n✅ Model saved to: models/best_model.joblib")
    print("✅ Results saved to: data/processed/")
    print("✅ Ready for Streamlit app!")


if __name__ == "__main__":
    main()
