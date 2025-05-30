# train_pipeline.py

import pickle
import pandas as pd
from model_pipeline import load_data, handle_missing_values, save_data

def main():
    # 1. Load
    df = load_data('data/raw/train.csv')
    # 2. Clean & fit
    df_clean, ids, mean_maps = handle_missing_values(df, mean_maps=None)
    # 3. Save cleaned train + ids
    save_data(df_clean, 'data/processed/clean_train.csv')
    pd.Series(ids, name='Id').to_csv('data/processed/id_train.csv', index=False)
    # 4. Persist mean_maps for test
    with open('data/processed/mean_maps.pkl','wb') as f:
        pickle.dump(mean_maps, f)

if __name__ == '__main__':
    main()
