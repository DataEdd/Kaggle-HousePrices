# test_pipeline.py

import pickle
import pandas as pd
from model_pipeline import load_data, handle_missing_values, save_data

def main():
    # 1. Load raw test & mean_maps
    df = load_data('data/raw/test.csv')
    with open('data/processed/mean_maps.pkl','rb') as f:
        mean_maps = pickle.load(f)
    # 2. Clean using train mean_maps
    df_clean, ids = handle_missing_values(df, mean_maps=mean_maps)
    # 3. Save cleaned test + ids
    save_data(df_clean, 'data/processed/clean_test.csv')
    pd.Series(ids, name='Id').to_csv('data/processed/id_test.csv', index=False)

if __name__ == '__main__':
    main()
