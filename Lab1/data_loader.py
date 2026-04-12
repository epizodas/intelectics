import pandas as pd
import os

def load_dataset(file_path):
    if not os.path.exists(file_path):
        print(f"Error: file '{file_path}' not found")
        return None

    try:
        df = pd.read_csv(file_path, index_col=0)
        print(f"Read {len(df)} entries")
        return df
    except Exception as e:
        print(f"Error: {e}")
        return None