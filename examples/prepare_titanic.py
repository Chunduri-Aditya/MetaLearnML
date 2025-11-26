"""
Helper script to prepare Titanic dataset for AutoML.
Run this after downloading Titanic from Kaggle.

Expected input: train.csv from Kaggle Titanic competition
Output: data/titanic_clean.csv with numeric features and integer labels
"""

import pandas as pd
from pathlib import Path


def prepare_titanic(
    input_path: str = "data/titanic_train.csv",
    output_path: str = "data/titanic_clean.csv"
):
    """
    Preprocess Titanic dataset for AutoML.
    
    Steps:
    1. Load raw CSV
    2. Drop non-predictive columns (PassengerId, Name, Ticket, Cabin)
    3. Encode categoricals (Sex, Embarked)
    4. Fill missing values
    5. Ensure numeric features and integer labels
    """
    print("Preparing Titanic dataset...")
    
    # Load data
    df = pd.read_csv(input_path)
    print(f"  Loaded {len(df)} samples")
    
    # Drop non-predictive columns
    cols_to_drop = ["PassengerId", "Name", "Ticket", "Cabin"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    print(f"  Dropped columns: {cols_to_drop}")
    
    # Encode Sex (binary: 0 = female, 1 = male)
    if "Sex" in df.columns:
        df["Sex_male"] = (df["Sex"] == "male").astype(int)
        df = df.drop(columns=["Sex"])
        print("  Encoded Sex -> Sex_male")
    
    # Encode Embarked (one-hot)
    if "Embarked" in df.columns:
        embarked_dummies = pd.get_dummies(df["Embarked"], prefix="Embarked")
        df = pd.concat([df, embarked_dummies], axis=1)
        df = df.drop(columns=["Embarked"])
        print("  Encoded Embarked -> one-hot")
    
    # Fill missing Age with median
    if "Age" in df.columns:
        df["Age"].fillna(df["Age"].median(), inplace=True)
        print("  Filled missing Age with median")
    
    # Fill missing Fare with median
    if "Fare" in df.columns:
        df["Fare"].fillna(df["Fare"].median(), inplace=True)
        print("  Filled missing Fare with median")
    
    # Ensure Survived is integer (0/1)
    if "Survived" in df.columns:
        df["Survived"] = df["Survived"].astype(int)
        print("  Ensured Survived is integer")
    
    # Get feature columns (everything except Survived)
    feature_cols = [c for c in df.columns if c != "Survived"]
    
    # Verify all features are numeric
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"  WARNING: {col} is not numeric, converting...")
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Save
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Saved cleaned dataset to {output_path}")
    print(f"  Features ({len(feature_cols)}): {feature_cols}")
    print(f"  Label column: Survived")
    print(f"  Samples: {len(df)}")
    print(f"  Classes: {df['Survived'].nunique()} ({df['Survived'].unique().tolist()})")
    print(f"  Class distribution: {df['Survived'].value_counts().to_dict()}")
    
    return df, feature_cols


if __name__ == "__main__":
    import sys
    
    input_file = sys.argv[1] if len(sys.argv) > 1 else "data/titanic_train.csv"
    
    if not Path(input_file).exists():
        print(f"ERROR: Input file not found: {input_file}")
        print("\nUsage:")
        print("  1. Download train.csv from Kaggle Titanic competition")
        print("  2. Place it in data/titanic_train.csv")
        print("  3. Run: python utils/prepare_titanic.py")
        print("\nOr specify custom path:")
        print("  python utils/prepare_titanic.py path/to/train.csv")
        sys.exit(1)
    
    prepare_titanic(input_file)

