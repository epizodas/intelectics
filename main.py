from data_loader import load_dataset
from analyzer import get_continuous_stats, get_categorical_stats
import pandas as pd

def main():
    file_path = './data/csgo.csv'
    print("Starting")

    df = load_dataset(file_path)
    
    if df is not None:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        
        print(get_continuous_stats(df).round(2))
        print()
        print(get_categorical_stats(df).round(2))

    else:
        print("error")

if __name__ == "__main__":
    main()