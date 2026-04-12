import pandas as pd

def get_continuous_stats(df):
    n_df = df.select_dtypes(include=['number'])
    stats_list = []

    ignored_columns = ["day", "month", "year", "team_a_rounds", "team_b_rounds", "rownames"]

    for col_name in n_df.columns:
        if(col_name in ignored_columns):
            continue

        col_data = n_df[col_name]
        
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

def remove_outliers(df):
    df_clean = df.copy()
    n_df = df_clean.select_dtypes(include=['number']).columns
    ignored_columns = ["day", "month", "year", "team_a_rounds", "team_b_rounds", "rownames"]

    for col_name in n_df:
        if(col_name in ignored_columns):
            continue
            
        Q1 = df_clean[col_name].quantile(0.25)
        Q3 = df_clean[col_name].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - IQR
        upper_bound = Q3 + IQR
        
        df_clean[col_name] = df_clean[col_name].clip(lower=lower_bound, upper=upper_bound)

    return df_clean

import pandas as pd

def normalize(df):
    n_df = df.select_dtypes(include=['number']).copy()

    ignored_columns = ["day", "month", "year", "team_a_rounds", "team_b_rounds", "rownames"]

    for col in n_df.columns:
        if col in ignored_columns:
            continue
        
        col_min = n_df[col].min()
        col_max = n_df[col].max()
        
        n_df[col] = (n_df[col] - col_min) / (col_max - col_min)

    return n_df

import pandas as pd