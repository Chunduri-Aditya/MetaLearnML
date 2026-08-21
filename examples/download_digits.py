import os

import pandas as pd
from sklearn.datasets import load_digits


def download_digits():
    print("Downloading Digits dataset...")
    digits = load_digits()
    df = pd.DataFrame(digits.data, columns=[f"pixel_{i}" for i in range(digits.data.shape[1])])
    df["target"] = digits.target

    save_path = os.path.join("data", "digits.csv")
    df.to_csv(save_path, index=False)
    print(f"Saved Digits dataset to {save_path}")
    print(f"Shape: {df.shape}")
    print(f"Classes: {sorted(df['target'].unique())}")


if __name__ == "__main__":
    download_digits()
