from data_loader import load_dataset
from analyzer import get_continuous_stats, get_categorical_stats, remove_outliers, normalize
from visualizer import plot_histograms, plot_categorical_bars, plot_boxplots, plot_splom, plot_correlation_examples, plot_grouped_histograms, plot_grouped_boxplots, plot_correlation_matrix
import pandas as pd

def main():
    file_path = '../data/csgo.csv'
    print("Starting")

    df = load_dataset(file_path)
    
    if df is not None:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        
        print(get_continuous_stats(df).round(2))
        print()
        print(get_categorical_stats(df).round(2))

        plot_histograms(df)
        plot_boxplots(df)

        df_cleaned = remove_outliers(df)
        print()
        print(get_continuous_stats(df_cleaned).round(2))
        
        plot_histograms(df_cleaned)
        plot_splom(df_cleaned)
        plot_correlation_examples(df_cleaned)
        plot_categorical_bars(df)
        plot_grouped_histograms(df_cleaned)
        plot_grouped_boxplots(df_cleaned)
        plot_correlation_matrix(df_cleaned)
        
        df_norm = normalize(df_cleaned)
        print()
        print(get_continuous_stats(df_norm).round(2))

    else:
        print("error")

if __name__ == "__main__":
    main()