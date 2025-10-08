#!/usr/bin/env python3
"""
Enhanced exoplanet model training pipeline.
This script processes the comprehensive dataset and trains an enhanced model.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}")
    print(f"Command: {command}")
    print("-" * 50)
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print("Stdout:", e.stdout)
        if e.stderr:
            print("Stderr:", e.stderr)
        return False

def main():
    """Main training pipeline."""
    print("🚀 Enhanced Exoplanet Model Training Pipeline")
    print("=" * 60)
    
    # Step 1: Enhanced data processing
    print("\n📊 Step 1: Enhanced Data Processing")
    success = run_command(
        "python src/data/enhanced_process_datasets.py",
        "Processing enhanced dataset with comprehensive features"
    )
    
    if not success:
        print("❌ Enhanced data processing failed!")
        return False
    
    # Step 2: Enhanced model training
    print("\n🤖 Step 2: Enhanced Model Training")
    success = run_command(
        "python src/models/enhanced_train_models.py",
        "Training enhanced models with full machine resources"
    )
    
    if not success:
        print("❌ Enhanced model training failed!")
        return False
    
    # Step 3: Verify outputs
    print("\n✅ Step 3: Verifying Outputs")
    
    enhanced_dataset = PROJECT_ROOT / "data" / "processed" / "enhanced_exoplanet_dataset.csv"
    enhanced_model = PROJECT_ROOT / "models" / "enhanced_best_model.joblib"
    enhanced_summary = PROJECT_ROOT / "models" / "enhanced_training_summary.json"
    
    if enhanced_dataset.exists():
        print(f"✅ Enhanced dataset: {enhanced_dataset}")
    else:
        print(f"❌ Enhanced dataset not found: {enhanced_dataset}")
    
    if enhanced_model.exists():
        print(f"✅ Enhanced model: {enhanced_model}")
    else:
        print(f"❌ Enhanced model not found: {enhanced_model}")
    
    if enhanced_summary.exists():
        print(f"✅ Training summary: {enhanced_summary}")
    else:
        print(f"❌ Training summary not found: {enhanced_summary}")
    
    print("\n🎉 Enhanced training pipeline completed!")
    print("\n📋 Next Steps:")
    print("1. Run the enhanced Streamlit app:")
    print("   streamlit run app/EnhancedStreamlitApp.py")
    print("2. The enhanced app will use the new model with 50+ features")
    print("3. Users can input comprehensive parameters for better accuracy")
    
    return True

if __name__ == "__main__":
    main()
