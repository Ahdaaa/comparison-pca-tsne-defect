# main.py

from src.loader import summarize_arff_datasets
from src.loader import load_and_merge_arff_datasets
from src.loader import load_single_arff_dataset

def run_summary():
    df = summarize_arff_datasets()
    print(df.to_string(index=False))
    return df

def merge_datasets():
    merged_df = load_and_merge_arff_datasets()
    print(merged_df.shape)
    return merged_df

def load_single(name):
    df = load_single_arff_dataset(name)
    print(df.shape)
    return df

if __name__ == "__main__":
    run_summary()
