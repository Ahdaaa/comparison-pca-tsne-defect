import os
import pandas as pd
from scipy.io import arff
from src.config import ARFF_FOLDER_PATH

def summarize_arff_datasets():
    dataset_summaries = []
    possible_label_names = {'defective', 'label', 'bug', 'class'}

    for filename in os.listdir(ARFF_FOLDER_PATH):
        if filename.endswith(".arff"):
            file_path = os.path.join(ARFF_FOLDER_PATH, filename)

            data, meta = arff.loadarff(file_path)
            df = pd.DataFrame(data)

            # Normalize column names
            df.columns = [col.decode().strip().lower() if isinstance(col, bytes) else col.strip().lower() for col in df.columns]
            dataset_name = filename.replace(".arff", "")

            # Try to detect the label column
            label_col = next((col for col in df.columns if col in possible_label_names), None)

            if not label_col:
                print(f"⚠️ No label column found in {filename}, skipping label stats.")
                num_defective = num_non_defective = "-"
            else:
                try:
                    # Decode if byte strings
                    if len(df) > 0 and isinstance(df[label_col].iloc[0], bytes):
                        df[label_col] = df[label_col].str.decode("utf-8")

                    unique_vals = set(df[label_col].unique())

                    # Map Y/N to 1/0
                    if unique_vals <= {"Y", "N"}:
                        df[label_col] = df[label_col].map({"Y": 1, "N": 0})

                    num_defective = int((df[label_col] == 1).sum())
                    num_non_defective = int((df[label_col] == 0).sum())
                except Exception as e:
                    print(f"⚠️ Failed to analyze label in {filename}: {e}")
                    num_defective = num_non_defective = "-"


            num_attributes = len(meta.names()) - 1  # exclude label
            num_instances = len(df)

            dataset_summaries.append({
                "Dataset": dataset_name,
                "Jumlah Atribut": num_attributes,
                "Jumlah Instance": num_instances,
                "Jumlah Defective": num_defective,
                "Jumlah Tidak Defective": num_non_defective,
                "Nama File": filename
            })

    df_summary = pd.DataFrame(dataset_summaries)
    df_summary = df_summary.sort_values("Dataset").reset_index(drop=True)
    return df_summary

def load_and_merge_arff_datasets():
    all_dfs = []
    common_columns = None
    possible_label_names = {'defective', 'label', 'bug', 'class'}  # Add more if needed
    final_label_name = 'defective'

    for filename in os.listdir(ARFF_FOLDER_PATH):
        if filename.endswith(".arff"):
            file_path = os.path.join(ARFF_FOLDER_PATH, filename)
            data, meta = arff.loadarff(file_path)

            df = pd.DataFrame(data)
            df.columns = [col.decode().strip().lower() if isinstance(col, bytes) else col.strip().lower() for col in df.columns]

            if len(df.columns) < 10:
                print(f"⚠️ Skipping {filename} — too few columns ({len(df.columns)})")
                continue

            # Try to detect label column
            label_col = next((col for col in df.columns if col in possible_label_names), None)

            if not label_col:
                print(f"⚠️ Skipping {filename} — no known label column found")
                continue

            # Rename label column to "defective"
            df.rename(columns={label_col: final_label_name}, inplace=True)

            # Change Y/N to 1/0
            # Decode byte string values in the label column
            if df[final_label_name].dtype == object and isinstance(df[final_label_name].iloc[0], bytes):
                df[final_label_name] = df[final_label_name].str.decode('utf-8')

            # Convert Y/N to 1/0
            if set(df[final_label_name].unique()) <= {'Y', 'N'}:
                df[final_label_name] = df[final_label_name].map({'Y': 1, 'N': 0})


            feature_columns = set(df.columns)

            print(f"{filename} includes label column: {label_col} → renamed to '{final_label_name}'")

            if common_columns is None:
                common_columns = feature_columns
            else:
                common_columns = common_columns.intersection(feature_columns)

            all_dfs.append(df)

    if not common_columns or final_label_name not in common_columns:
        raise ValueError(f"No common columns found including label '{final_label_name}'.")

    print("✅ Final common columns:", sorted(common_columns))

    selected_columns = list(common_columns)
    merged_df = pd.concat([df[selected_columns] for df in all_dfs], ignore_index=True)

    return merged_df

def load_single_arff_dataset(filename):
    import os
    import pandas as pd
    from scipy.io import arff

    possible_label_names = {'defective', 'label', 'bug', 'class'}
    final_label_name = 'defective'

    file_path = os.path.join(ARFF_FOLDER_PATH, filename)

    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)

    # Normalisasi nama kolom
    df.columns = [col.decode().lower().strip() if isinstance(col, bytes) else col.lower().strip() for col in df.columns]

    # Deteksi label
    label_col = next((col for col in df.columns if col in possible_label_names), None)
    if not label_col:
        raise ValueError(f"Tidak ditemukan label di kolom: {df.columns.tolist()}")

    # Rename kolom label ke 'defective'
    df.rename(columns={label_col: final_label_name}, inplace=True)

    # Decode dan map Y/N ke 1/0 jika diperlukan
    if df[final_label_name].dtype == object and isinstance(df[final_label_name].iloc[0], bytes):
        df[final_label_name] = df[final_label_name].str.decode("utf-8")
    if set(df[final_label_name].unique()) <= {'Y', 'N'}:
        df[final_label_name] = df[final_label_name].map({'Y': 1, 'N': 0})

    return df
