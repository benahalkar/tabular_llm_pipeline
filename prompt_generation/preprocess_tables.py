import json
import yaml
import numpy as np 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder
import os
from datetime import datetime


def load_metadata(metadata_file_path):
    """Load the metadata from metadata.json file"""
    with open(metadata_file_path, 'r') as f:
        return json.load(f)


def load_config(config_file_path):
    """Load the configuration file"""
    with open(config_file_path, 'r') as f:
        return yaml.safe_load(f)


def read_table(table_name, raw_data_path, is_synthetic):
    """
    Read a table file based on whether it's synthetic (CSV) or not (parquet)
    
    Args:
        table_name (str): Name of the table file
        is_synthetic (bool): Whether the table is synthetic (CSV) or not (parquet)
        
    Returns:
        pd.DataFrame: The loaded dataframe
    """

    if is_synthetic:
        return pd.read_csv(raw_data_path + table_name, header=None)
    else:
        return pd.read_parquet(raw_data_path + table_name)
    

def adjust_columns(table_info, df, target_idx, max_cols, num_cols):
    """
    Adjust the number of columns in the dataframe to match max_cols.
    If current columns < max_cols: pad with zeros
    If current columns > max_cols: randomly drop columns
    
    Args:
        df (pd.DataFrame): Input dataframe (features only, no target)
        max_cols (int): Target number of columns
        num_cols (int): Original number of columns from metadata
        
    Returns:
        pd.DataFrame: Adjusted dataframe
    """
    # print("Check 1ca")
    # print(f"Num Cols: {num_cols}")
    # print(f"Max Cols: {max_cols}")

    # If current columns > max_cols, randomly drop columns
    if num_cols > max_cols:
        # print("Check 1cb1")
        # Randomly select columns to keep
        feature_cols = list(range(num_cols))
        feature_cols.remove(table_info['label_index'])
        cols_to_keep = np.random.choice(
            feature_cols, 
            size=max_cols-1,  # -1 because we already removed target
            replace=False
        )
        cols_to_keep = np.sort(cols_to_keep)

        table_info['num_cols'] = max_cols
        # print(table_info)
        # print(cols_to_keep)
        # print(len(table_info['categorical_columns']))

        retained_categorical_columns = [table_info['categorical_columns'][i] for i in cols_to_keep]
        table_info['categorical_columns'] = retained_categorical_columns
        # print("1cb2")

        return table_info, df.iloc[:, cols_to_keep]
    
    # If current columns < max_cols, pad with zeros
    elif num_cols <= max_cols-1:
        # print("Check 1cb2")
        padding_cols = {f'padded_{i}': 0 for i in range(num_cols, max_cols)}
        padding_df = pd.DataFrame(0, index=df.index, columns=padding_cols.keys())
        feature_df = df.drop(df.columns[target_idx], axis=1)
        # print(padding_df)
        _ = table_info['categorical_columns'].pop(target_idx)
        return table_info, pd.concat([feature_df, padding_df], axis=1)
    
    return table_info, df.drop(df.columns[target_idx], axis=1)


def preprocess_table(table_info, raw_data_path, table_encoder_type='MLP', max_cols=None, apply_mask=False, 
                     mask_fraction=0.1, mask_token='missing', decimal_value_count=2, max_rows_cols=25000,
                     is_training_encoder=False):
    """
    Preprocess a single table based on the model type and categorical indices
    
    Args:
        table_info (dict): Dictionary containing table metadata
        model_type (str): Type of model (MLP or other)
        apply_mask (bool): Whether to apply masking
        mask_fraction (float): Fraction of target values to mask
        max_cols (int): Maximum number of columns for padding
    Returns:
        tuple: (processed_df, updated_table_info)
    """
    table_name = table_info['table_name']
    is_synthetic = table_info['is_synthetic_data']
    num_cols = table_info['num_cols'] 
    
    table_info['target_masked'] = False
    table_info['masked_info'] = {}

    # print(table_info)

    df = read_table(table_name, raw_data_path, is_synthetic)
    
    # print("Check 1a")
    target_idx = table_info['label_index']
    target_col = df.iloc[:, target_idx]

    # print("Check 1b")
    if table_encoder_type.upper() == 'MLP':

        # print("Check 1c")

        if not is_synthetic and max_cols is not None:
            is_target_categorical = table_info['categorical_columns'][target_idx]
            table_info, feature_df = adjust_columns(table_info, df, target_idx, max_cols, num_cols)

            # print(feature_df.shape)
            # print(len(table_info['categorical_columns']))
            encoded_feature_df = feature_df.copy()
            for idx, is_categorical in enumerate(table_info['categorical_columns']):
                if is_categorical == 1 and idx < len(feature_df.columns):
                    if feature_df[feature_df.columns[idx]].dtype == 'bool':
                        encoded_feature_df[encoded_feature_df.columns[idx]] = feature_df[feature_df.columns[idx]].astype(int)
                    else:
                        le = LabelEncoder()
                        encoded_feature_df[encoded_feature_df.columns[idx]] = le.fit_transform(feature_df[feature_df.columns[idx]])
            feature_df = encoded_feature_df.copy()
            del encoded_feature_df
                    
            if is_target_categorical:
                le = LabelEncoder()
                target_col = le.fit_transform(target_col)
            else:
                if target_col.dtype == 'bool':
                    target_col = target_col.astype(int)
        
        if is_synthetic:
            feature_df = df.drop(df.columns[target_idx], axis=1)

    
    if table_encoder_type.upper() != 'MLP':
        # print("Check 1d")
        feature_df = df.drop(df.columns[target_idx], axis=1)
        if not is_synthetic:
            is_target_categorical = table_info['categorical_columns'].pop(target_idx)
        else:
            feature_df = feature_df.iloc[:, :num_cols-1]

        if apply_mask:
            mask_indices = np.random.choice(len(target_col),
                                          size=int(len(target_col) * mask_fraction),
                                          replace=False)
            # print(mask_indices)
            # Store original values
            masked_info = {str(idx): str(target_col.iloc[idx]) for idx in mask_indices}
            target_col = target_col.astype(str)
            target_col.iloc[mask_indices] = mask_token
            # print("Check 1e")
            table_info['target_masked'] = True
            table_info['masked_info'] = masked_info
    
    if is_synthetic:
        result_df = pd.concat([feature_df, target_col], axis=1)
        # print("1f")
    else:
        result_df = feature_df.copy()
        result_df.insert(feature_df.shape[1], df.columns[target_idx], target_col)
        table_info['categorical_columns'].insert(feature_df.shape[1], is_target_categorical)
    table_info['label_index'] = feature_df.shape[1]

    # print("1g")
    result_df = result_df.map(lambda x: round(x, decimal_value_count) if isinstance(x, float) else x)
    # print("1h")
    max_rows_retained = int(np.ceil(max_rows_cols/result_df.shape[1]))
    if (not is_training_encoder) and (max_rows_retained < result_df.shape[0]):
        result_df = result_df.sample(n=max_rows_retained, random_state=11)
        table_info['num_rows'] = max_rows_retained

    if table_name.endswith(".parquet"):
        table_name = table_name.replace(".parquet", ".csv")
        table_info['table_name'] = table_name
    
    return result_df, table_info

def main():
    preprocessing_start_time = datetime.now() 
    # Load metadata and config
    config_file_path = 'config/preprocess_config.yaml'
    config = load_config(config_file_path)
    
    metadata_path = config['data_settings']['raw_metadata_file']
    if not metadata_path:
        raise ValueError("metadata_path not found in config file")
    
    metadata = load_metadata(metadata_path)
    processed_metadata = []

    print(f"Total Number of Tables: {len(metadata)}")
    
    is_training_encoder = config['data_settings']['is_training_encoder']
    table_encoder_type = config['data_settings']['table_encoder_type']
    max_cols = config['data_settings'].get('max_cols')
    apply_mask = config['data_settings'].get('apply_target_mask', False)
    mask_fraction = config['data_settings'].get('mask_fraction', 0.0)
    mask_token = config['data_settings'].get('mask_token', 'missing')
    raw_data_path = config['data_settings'].get('raw_data_path')
    processed_data_path = config['data_settings'].get('processed_data_path')
    processed_metadata_path = config['data_settings'].get('processed_metadata_file')
    decimal_value_count = config['data_settings'].get('decimal_value_count')
    max_token_count_per_cell = config['data_settings'].get('max_token_count_per_cell')
    total_context_length = config['data_settings'].get('total_context_length')
    max_rows_cols = total_context_length/max_token_count_per_cell
    
    os.makedirs(processed_data_path, exist_ok=True)
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)  # Create parent Data directory if needed
    
    for idx, table_info in enumerate(metadata):
        table_name = table_info['table_name']

        if table_info['label_index'] == -1:
            continue
        
        if not os.path.exists(raw_data_path + table_name):
            print(f"Warning: Table {table_name} not found, skipping...")
            continue
        
        try:
            # print("Check 1")
            processed_df, updated_table_info = preprocess_table(
                table_info, 
                raw_data_path,
                table_encoder_type=table_encoder_type,
                max_cols=max_cols,
                apply_mask=apply_mask,
                mask_fraction=mask_fraction,
                mask_token=mask_token,
                decimal_value_count=decimal_value_count,
                max_rows_cols=max_rows_cols,
                is_training_encoder=is_training_encoder
            )
            # print("Check 2")
            processed_metadata.append(updated_table_info)
            processed_table_path = os.path.join(processed_data_path, updated_table_info['table_name'])
            processed_df.to_csv(processed_table_path, index=False)
            
            print(f"Successfully preprocessed {table_name}")
            
        except Exception as e:
            print(f"Error processing {table_name}: {str(e)}")
    
    with open(processed_metadata_path, 'w') as f:
        json.dump(processed_metadata, f, indent=4)
    
    preprocessing_end_time = datetime.now()
    print(f"Total Preprocessing Time: {preprocessing_end_time - preprocessing_start_time}")  # Print the execution time


if __name__ == "__main__":
    main() 