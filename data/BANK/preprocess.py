import numpy as np
import pandas as pd

def divide_bank():
    df = pd.read_csv('bank-additional-full.csv')
    # df = df.sample(frac=1.0).reset_index(drop=True)

    # split_idx = int(df.shape[0] * 0.8)
    split_idx = 10000
    cols = list(df.columns)
    target_idx = cols.index("y")
    cols = cols[target_idx+1:] + cols[:target_idx+1]
    df = df[cols]
    train_data = df[split_idx:]
    test_data = df[:split_idx]

    train_data.to_csv("train_data.csv", index=False)
    test_data.to_csv("test_data.csv", index=False)

if __name__ == '__main__':
    divide_bank()