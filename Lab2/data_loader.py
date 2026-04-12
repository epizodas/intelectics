import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

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
    
def prepare_data_for_tree(df):
    counts = df['map'].value_counts()
    valid_maps = counts[counts >= 5].index
    df = df[df['map'].isin(valid_maps)].copy()
    
    print(f"Pašalinti reti žemėlapiai. Likę žemėlapiai: {df['map'].nunique()}")
    print(df['map'].value_counts())

    ignored_columns = ["day", "month", "year", "team_a_rounds", "team_b_rounds", "rownames", "date"]
    
    y = df['map']
    X = df.drop(columns=['map'] + ignored_columns, errors='ignore')

    if 'result' in X.columns:
        le_result = LabelEncoder()
        X['result'] = le_result.fit_transform(X['result'])

    return X, y