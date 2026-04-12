import matplotlib.pyplot as plt
import math
import pandas as pd
import seaborn as sns
import numpy as np

def plot_histograms(df):
    n_df = df.select_dtypes(include=['number'])
    cols = n_df.columns.tolist()
    ignored_columns = ["day", "month", "year", "team_a_rounds", "team_b_rounds", "rownames"]
    
    cols_to_plot = [col for col in cols if col not in ignored_columns]    
    
    n = len(df)
    k = 1 + 3.232 * math.log10(n)
    bins_count = round(k) 
    print(bins_count)
    
    cols_per_row = 3
    rows_count = 3

    fig, axes = plt.subplots(rows_count, cols_per_row, figsize=(15, 4 * rows_count))
    axes = axes.flatten()

    for i, col_name in enumerate(cols_to_plot):
        ax = axes[i]
        col_data = n_df[col_name].dropna()
        
        bins = bins_count
        if (col_data.nunique() < bins_count):
            bins = col_data.nunique()

        ax.hist(col_data, bins=bins, color='navy', edgecolor='black', alpha=0.7)
        ax.set_title(col_name)
        ax.set_xlabel('Reikšmė')
        ax.set_ylabel('Dažnis')
        ax.grid(axis='y', alpha=0.5)

    plt.tight_layout(rect=(0, 0.03, 1, 0.95)) 
    plt.show()

def plot_categorical_bars(df):
    c_df = df.select_dtypes(include=['object'])
    cols = c_df.columns.tolist()
    ignored_columns = ["date", "rownames"]
    
    cols_to_plot = [col for col in cols if col not in ignored_columns]
    
    cols_per_row = 2
    rows_count = 1
    
    fig, axes = plt.subplots(rows_count, cols_per_row, figsize=(15, 5 * rows_count))
    axes = axes.flatten()

    for i, col_name in enumerate(cols_to_plot):
        ax = axes[i]
        col_data = df[col_name].value_counts()
        ax.bar(col_data.index, col_data.values, color='salmon', edgecolor='black', alpha=0.7)
        ax.set_title(col_name)
        ax.set_ylabel('Kiekis')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.5)

    plt.tight_layout(rect=(0, 0.03, 1, 0.95)) 
    plt.show()

def plot_boxplots(df):
    numeric_df = df.select_dtypes(include=['number'])
    all_numeric_cols = numeric_df.columns.tolist()
    
    ignored_columns = ["day", "month", "year", "team_a_rounds", "team_b_rounds", "rownames"]
    cols_to_plot = [col for col in all_numeric_cols if col not in ignored_columns]
    
    cols_per_row = 3  
    rows_count = 3
    
    fig, axes = plt.subplots(rows_count, cols_per_row, figsize=(16, 5 * rows_count))
    
    axes = axes.flatten()

    for i, col_name in enumerate(cols_to_plot):
        ax = axes[i]
        data = numeric_df[col_name].dropna()
        ax.boxplot(data, vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightgreen', color='black'),
                   medianprops=dict(color='red', linewidth=1.5),
                   flierprops=dict(marker='o', markerfacecolor='red', markersize=5))
        ax.set_title(col_name)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_xticks([])

    plt.tight_layout(rect=(0, 0.03, 1, 0.95)) 
    plt.show()

def plot_splom(df):
    numeric_df = df.select_dtypes(include=['number'])
    
    ignored_columns = ["day", "month", "year", "team_a_rounds", "team_b_rounds", "rownames"]
    
    df_plot = numeric_df.drop(columns=ignored_columns, errors='ignore')
    
    sns.set_theme(style="ticks")
    sns.pairplot(df_plot, 
                     corner=True, 
                     kind='scatter', 
                     diag_kind='auto',
                     plot_kws={'alpha': 0.6, 's': 15, 'edgecolor': 'none'},
                     height=2.5)
    
    plt.show()

def plot_correlation_examples(df):
    pairs = [
        ('kills', 'points'),
        ('match_time_s', 'deaths'),
        ('wait_time_s', 'kills'),
        ('ping', 'hs_percent')
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, (col_x, col_y) in enumerate(pairs):
        ax = axes[i]
        x = df[col_x]
        y = df[col_y]
        corr_coef = x.corr(y)
        ax.scatter(x, y, alpha=0.6, c='royalblue', edgecolors='none')

        if len(x) > 1:
            try:
                m, b = np.polyfit(x, y, 1)
                ax.plot(x, m*x + b, color='red', linewidth=1.5, linestyle='--')
            except:
                pass

        ax.set_title(f"\nKoreliacija r = {corr_coef:.2f}")
        ax.set_xlabel(col_x)
        ax.set_ylabel(col_y)
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.show()

def plot_grouped_histograms(df):    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.histplot(data=df, x='points', hue='result', kde=True, element="step", ax=axes[0], palette='Set1')
    axes[0].set_xlabel('points')
    
    sns.histplot(data=df, x='deaths', hue='result', kde=True, element="step", ax=axes[1], palette='Set1')
    axes[1].set_xlabel('deaths')

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.show()

def plot_grouped_boxplots(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    sns.boxplot(data=df, x='map', y='wait_time_s', ax=axes[0], palette='coolwarm')
    axes[0].set_xlabel('map')
    axes[0].set_ylabel('wait_time_s (s)')
    axes[0].tick_params(axis='x', rotation=45)
    
    sns.boxplot(data=df, x='result', y='kills', ax=axes[1], palette='coolwarm')
    axes[1].set_xlabel('result')
    axes[1].set_ylabel('kills')

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.show()

def plot_correlation_matrix(df):
    numeric_df = df.select_dtypes(include=['number'])
    ignored_columns = ["day", "month", "year", "team_a_rounds", "team_b_rounds", "rownames"]
    df_plot = numeric_df.drop(columns=ignored_columns, errors='ignore')
    
    if df_plot.empty:
        return

    corr_matrix = df_plot.corr()
    
    plt.figure(figsize=(12, 10))

    sns.heatmap(corr_matrix, 
                annot=True, 
                fmt=".2f", 
                cmap='coolwarm', 
                center=0, 
                square=True, 
                linewidths=0.5, 
                cbar_kws={"shrink": 0.8})
    
    plt.tight_layout()
    plt.show()