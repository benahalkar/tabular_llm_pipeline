import os
import pandas as pd
import json
import random
import numpy as np
import re
import yaml

def extract_causal_table_number(file_name):
    match = re.search(r'\d+', file_name)  # Look for one or more digits in the string
    if match:
        return int(match.group())  # Convert the match to an integer
    return None

def select_label_column(df):
    """
    Select the label column for real data based on heuristics and probabilities.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
    
    Returns:
        label_index (int): Index of the selected label column.
        num_classes (int): Number of unique values in the selected label column (if categorical).
    """

    original_columns = df.columns.tolist()
    
    # Drop columns whose names start with "Unnamed:"
    df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]
 
    # Drop columns with all unique values
    valid_cols = [col for col in df.columns if df[col].nunique() != len(df)]
    
    # Drop date columns
    valid_cols = [
        col for col in valid_cols 
        if not pd.api.types.is_datetime64_any_dtype(df[col]) and not pd.api.types.is_timedelta64_dtype(df[col])
    ]
    
    # Drop columns with overly long serialized values
    valid_cols = [
        col for col in valid_cols 
        if df[col].astype(str).map(len).max() <= 256
    ]

    # If no valid columns, return -1 as label_index and 0 for num_classes
    if not valid_cols:
        return -1, 0
    
    # Classify remaining columns as categorical or continuous
    categorical_candidates = [
        col for col in valid_cols if isinstance(df[col].dtype, pd.CategoricalDtype) or df[col].nunique() < 0.1 * len(df)
    ]
    continuous_candidates = [
        col for col in valid_cols if col not in categorical_candidates
    ]
    
    # Choose label column based on probabilities
    if categorical_candidates and (not continuous_candidates or random.random() < 0.9):
        chosen_col = random.choice(categorical_candidates)
    elif continuous_candidates:
        chosen_col = random.choice(continuous_candidates)
    else:
        # Default case: choose any valid column
        valid_cols = df.columns.tolist()  # Ensure to get the remaining columns
        chosen_col = random.choice(valid_cols)
    
    label_index = original_columns.index(chosen_col)
    num_classes = df[chosen_col].nunique() if chosen_col in categorical_candidates else 0
    return label_index, num_classes


def generate_metadata_with_label_selection(folder_path: str, causal_metadata_path: str = "causal_metadata.json", metadata_output_path: str = "metadata.json"):
    """
    Generates metadata for all tables (CSV and Parquet) in a folder, with label column heuristics applied only to real data.
    
    Parameters:
        folder_path (str): Path to the folder containing CSV and Parquet files.
        metadata_output (str): Path to save the metadata JSON file.
    """
    metadata = []
    with open(causal_metadata_path, 'r') as causal_metadata_file:
        causal_metadata = json.load(causal_metadata_file)

    
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        try:
            has_causal = False
            causal_info = []
            # Determine if the file is synthetic data or real data
            if file_name.endswith(".csv"):
                df = pd.read_csv(file_path)
                synthetic_data = True

                if file_name.startswith("Causal_Table"):
                    has_causal = True
                    causal_entry = extract_causal_table_number(file_name) - 1
                    causal_info = causal_metadata[causal_entry]['causal_info']
                                
                # Metadata extraction for synthetic data
                table_name = file_name
                num_rows = len(df)
                num_cols = int((df.any(axis=0)).sum().item())  
                categorical_columns = [0] * num_cols  # All columns are numerical
                label_index = len(df.columns) - 1  # Last column is the label
                num_classes = df.iloc[:, -1].nunique()  # Unique values in the last column (label)
                classification = True
            
            elif file_name.endswith(".parquet"):
                df = pd.read_parquet(file_path)
                synthetic_data = False
                
                # Metadata extraction for real data
                table_name = file_name
                num_rows = len(df)  # Adjust for real data
                num_cols = len(df.columns)
                categorical_columns = [
                    1 if pd.api.types.is_bool_dtype(df[col]) else 0 if pd.api.types.is_numeric_dtype(df[col]) else 1 
                    for col in df.columns
                ]
                
                # Apply label selection heuristics
                label_index, num_classes = select_label_column(df)
                classification = True if num_classes!=0 else False
            
            else:
                continue  # Skip files that are neither CSV nor Parquet
            
            metadata.append({
                "table_name": table_name,
                "is_synthetic_data": synthetic_data,
                "has_causal": has_causal,
                "causal_info": causal_info,
                "num_rows": num_rows,
                "num_cols": num_cols,
                "categorical_columns": categorical_columns,
                "label_index": label_index,
                "num_classes": num_classes,
                'is_classification': classification
            })
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
    
    # Save metadata to JSON
    with open(metadata_output_path, "w") as json_file:
        json.dump(metadata, json_file, indent=4)
    print(f"Metadata saved to {metadata_output_path}")


def main():
    config_file_path = 'config/preprocess_config.yaml'
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)

    raw_data_path = config['data_settings'].get('raw_data_path')
    casual_metadata_path = config['data_settings'].get('causal_metadata_path')
    metadata_output_path = config['data_settings'].get('raw_metadata_file')
    generate_metadata_with_label_selection(raw_data_path, casual_metadata_path, metadata_output_path)


if __name__ == "__main__":
    main()