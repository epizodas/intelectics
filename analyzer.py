import pandas as pd

def get_continuous_stats(df):
    numeric_df = df.select_dtypes(include=['number'])
    stats_list = []

    for col_name in numeric_df.columns:
        if(col_name == "day" or col_name == "month" or col_name == "year" or col_name == "team_a_rounds" or col_name == "team_b_rounds"):
            continue

        col_data = numeric_df[col_name]
        
        stats = {
            'Atributas': col_name,
            'Kardinalumas': col_data.nunique(),
            'Min': col_data.min(),
            'Q1 (25%)': col_data.quantile(0.25),
            'Mediana': col_data.median(),
            'Vidurkis': col_data.mean(),
            'Q3 (75%)': col_data.quantile(0.75),
            'Max': col_data.max(),
            'Std. nuokrypis': col_data.std()
        }
        stats_list.append(stats)
    
    results_df = pd.DataFrame(stats_list)
    results_df.set_index('Atributas', inplace=True)
    
    return results_df

def get_categorical_stats(df):
    cat_df = df.select_dtypes(include=['object'])
    
    stats_list = []

    for col_name in cat_df.columns:
        if(col_name == "date"):
            continue
        col_data = cat_df[col_name]
        total_rows = len(df)
        
        val_counts = col_data.value_counts()
        
        mode1_val = val_counts.index[0]
        mode1_freq = val_counts.iloc[0]
        mode2_val = val_counts.index[1]
        mode2_freq = val_counts.iloc[1]

        stats = {
            'Atributas': col_name,
            'Kardinalumas': col_data.nunique(),
            'Moda(1)': mode1_val,
            'Modos(1) dažnumas': mode1_freq,
            'Modos(1) %': (mode1_freq / total_rows) * 100,
            'Moda(2)': mode2_val,
            'Modos(2) dažnumas': mode2_freq,
            'Modos(2) %': (mode2_freq / total_rows) * 100,
        }
        
        stats_list.append(stats)
    
    results_df = pd.DataFrame(stats_list)
    results_df.set_index('Atributas', inplace=True)
    
    return results_df