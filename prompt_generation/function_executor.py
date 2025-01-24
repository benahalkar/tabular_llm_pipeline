import random
import numpy as np

def execute_function(func_name, table_data, functions, categorical_columns, label_index, table_encoder_type, num_cols, num_rows):
    """
    Execute a function with random parameters based on table dimensions.
    Returns parameters and result.
    """
    func = functions[func_name]
    params = {}
    numerical_column_indices, categorical_column_indices= [], []
    for i, value in enumerate(categorical_columns):
        if i == label_index: #Dont include label column
                continue
        if value == 0:
            numerical_column_indices.append(i)
        else:
            categorical_column_indices.append(i)
    
    def get_random_axis_and_index():
        axis = random.choice([0, 1])
        max_index = num_rows - 1 if axis == 0 else num_cols - 1
        return axis, random.randint(0, max_index)

    def get_numerical_column_index(numerical_column_indices):
        axis = 1
        index = random.choice(numerical_column_indices) if numerical_column_indices else None
        return axis, index

    def get_random_slice(lst, slice_size=5):
        """Return a random slice of the list with specified size and indices."""
        if len(lst) <= slice_size:
            return lst, 0, len(lst)
        
        max_start = len(lst) - slice_size
        start_idx = random.randint(0, max_start)
        end_idx = start_idx + slice_size
        
        return lst[start_idx:end_idx], start_idx, end_idx

    def get_random_cell():
        return (random.randint(0, num_rows - 1), random.randint(0, num_cols - 1))
    
    def get_random_numeric_cell(numerical_column_indices):
        return (random.randint(0, num_rows - 1), random.choice(numerical_column_indices) if numerical_column_indices else None)

    if func_name == 'find_value':
        random_cell = get_random_cell()
        params = {'row': random_cell[0], 'col': random_cell[1], 'result': func(table_data, random_cell[0], random_cell[1])}

    elif func_name in ['sum_values', 'value_range', 'mean_value', 'median_value', 
                    'mode_value', 'count_empty_cells']:
        axis, index = get_numerical_column_index(numerical_column_indices)
        if index is None:
            params = None
        else:
            params = {
                'index': index,
                'axis': axis,
                'axis_name': 'row' if axis == 0 else 'column',
                'result': func(table_data, index, axis)
            }

    elif func_name in ['square_values', 'sqrt_values', 'abs_values']:
        axis, index = get_numerical_column_index(numerical_column_indices)
        if index is None:
            params = None
        else:
            full_result = func(table_data, index, axis).tolist()
            result_slice, slice_start, slice_end = get_random_slice(full_result)
            params = {
                'index': index,
                'axis': axis,
                'axis_name': 'row' if axis == 0 else 'column',
                'result': result_slice,
                'slice_start': slice_start,
                'slice_end': slice_end
            }
    
    elif func_name in ['non_empty_cells', 'unique_values']:
        axis, index = get_random_axis_and_index()
        full_result = func(table_data, index, axis).tolist()
        result_slice, slice_start, slice_end = get_random_slice(full_result)
        params = {
            'index': index,
            'axis': axis,
            'axis_name': 'row' if axis == 0 else 'column',
            'result': result_slice,
            'slice_start': slice_start,
            'slice_end': slice_end
        }

    elif func_name == 'has_missing_values':
        axis, index = get_random_axis_and_index()
        result = func(table_data, index, axis)
        params = {
            'index': index,
            'axis': axis,
            'axis_name': 'row' if axis == 0 else 'column',
            'presence': 'are' if result else 'are no'
        }

    elif func_name == 'compare_rows_or_columns':
        axis, index1 = get_random_axis_and_index()
        index2 = random.choice(range(table_data.shape[axis]))
        result = func(table_data, index1, index2, axis)
        params = {
            'index1': index1,
            'index2': index2,
            'axis': axis,
            'axis_name': 'row' if axis == 0 else 'column',
            'presence': '' if result else ' not'
        }

    elif func_name == 'value_present':
        axis, index = get_random_axis_and_index()
        value = random.choice(table_data.values.flatten().tolist())
        result = func(table_data, index, value, axis)
        params = {
            'index': index,
            'axis': axis,
            'axis_name': 'row' if axis == 0 else 'column',
            'value': value,
            'presence': '' if result else ' not'
        }

    elif func_name == 'max_min_value':
        axis, index = get_numerical_column_index(numerical_column_indices)
        if index is None:
            params = None
        else:
            max_val, min_val = func(table_data, index, axis)
            params = {
                'index': index,
                'axis': axis,
                'axis_name': 'row' if axis == 0 else 'column',
                'max_val': max_val,
                'min_val': min_val
            }

    elif func_name == 'cumulative_sum':
        axis, index = get_numerical_column_index(numerical_column_indices)
        if index is None:
            params = None
        else:
            full_result = func(table_data, index, axis).tolist()
            result_slice, slice_start, slice_end = get_random_slice(full_result)
            params = {
                'index': index,
                'axis': axis,
                'axis_name': 'row' if axis == 0 else 'column',
                'result': result_slice,
                'slice_start': slice_start,
                'slice_end': slice_end
            }

    elif func_name == 'successive_differences':
        axis, index = get_numerical_column_index(numerical_column_indices)
        if index is None:
            params = None
        else:
            full_result = func(table_data, index, axis).dropna().tolist()
            result_slice, slice_start, slice_end = get_random_slice(full_result)
            params = {
                'index': index,
                'axis': axis,
                'axis_name': 'row' if axis == 0 else 'column',
                'result': result_slice,
                'slice_start': slice_start,
                'slice_end': slice_end
            }

    elif func_name == 'fetch_neighbors':
        random_cell = get_random_cell()
        params = {'row': random_cell[0], 'col': random_cell[1], 'result': func(table_data, random_cell[0], random_cell[1])}

    elif func_name in ['top_n_values', 'bottom_n_values']:
        axis, index = get_numerical_column_index(numerical_column_indices)
        if index is None:
            params = None
        else:
            n = random.randint(1, 15)  
            params = {
                'index': index,
                'axis': axis,
                'axis_name': 'row' if axis == 0 else 'column',
                'n': n,
                'result': func(table_data, index, n, axis).tolist()
            }

    elif func_name == 'sort_values':
        axis, index = get_numerical_column_index(numerical_column_indices)
        if index is None:
            params = None
        else:
            ascending = random.choice([True, False])
            full_result = func(table_data, index, ascending, axis).tolist()
            result_slice, slice_start, slice_end = get_random_slice(full_result)
            params = {
                'index': index,
                'axis': axis,
                'axis_name': 'row' if axis == 0 else 'column',
                'order': 'ascending' if ascending else 'descending',
                'result': result_slice,
                'slice_start': slice_start,
                'slice_end': slice_end
            }

    elif func_name in ['add_cells', 'subtract_cells']:
        cell1 = get_random_numeric_cell(numerical_column_indices)
        cell2 = get_random_numeric_cell(numerical_column_indices)
        if (cell1[1] is None) or (cell2[1] is None):
            params = None
        else:
            params = {
                'cell1': cell1,
                'cell2': cell2,
                'result': func(table_data, cell1, cell2)
            }

    elif func_name == 'add_constant':
        axis, index = get_numerical_column_index(numerical_column_indices)
        if index is None:
            params = None
        else:
            constant = random.uniform(1, 100)
            full_result = func(table_data, index, constant, axis).tolist()
            result_slice, slice_start, slice_end = get_random_slice(full_result)
            params = {
                'index': index,
                'axis': axis,
                'axis_name': 'row' if axis == 0 else 'column',
                'constant': constant,
                'result': result_slice,
                'slice_start': slice_start,
                'slice_end': slice_end
            }

    elif func_name == 'table_shape':
        shape = func(table_data)
        params = {'rows': shape[0], 'cols': shape[1]}

    elif func_name == 'fill_missing_values':
        axis, index = get_random_axis_and_index()
        value = random.randint(0, 100)
        full_result = func(table_data, index, value, axis).tolist()
        result_slice, slice_start, slice_end = get_random_slice(full_result)
        params = {
            'index': index,
            'axis': axis,
            'axis_name': 'row' if axis == 0 else 'column',
            'value': value,
            'result': result_slice,
            'slice_start': slice_start,
            'slice_end': slice_end
        }

    elif func_name in ['find_all_cells_with_value', 'count_value_occurrences']:
        table_non_zero_data = set(table_data.values.flatten().tolist()) - {0.0}
        value = random.choice(list(table_non_zero_data))
        params = {'value': value, 'result': func(table_data, value)}

    elif func_name in ['sum_table_values', 'average_table_values']:
        if table_encoder_type != 'MLP':
            params = None
        else:
            params = {'result': func(table_data)}
    
    elif func_name == 'direct_correlation_between_columns':
        if numerical_column_indices and len(numerical_column_indices) >= 2:
            col1, col2 = random.sample(numerical_column_indices, 2)
            correlation = func(table_data, col1, col2)
            if (correlation is not None) and (not np.isnan(correlation)):
                params = {
                    'col1': col1,
                    'col2': col2,
                    'correlation': round(correlation, 3)
                }
            else:
                params = None
        else:
            params = None
    
    elif func_name == 'stepwise_correlation_between_columns':
        if numerical_column_indices and len(numerical_column_indices) >= 2:
            col1, col2 = random.sample(numerical_column_indices, 2)
            result = func(table_data, col1, col2)
            
            if (result['correlation'] is not None) and (not np.isnan(result['correlation'])):
                params = {
                    'col1': col1,
                    'col2': col2,
                    'correlation': round(result['correlation'], 3),
                    'x_mean': round(result['x_mean'], 3),
                    'y_mean': round(result['y_mean'], 3),
                    'n_samples': result['n_samples'],
                    'sum_product_dev': round(result['sum_product_dev'], 3),
                    'sum_squared_dev_x': round(result['sum_squared_dev_x'], 3),
                    'sum_squared_dev_y': round(result['sum_squared_dev_y'], 3)
                }
            else:
                params = None
        else:
            params = None
    
    elif func_name in ['transform_column', 'stepwise_transform_column']:
        if numerical_column_indices:
            col = random.choice(numerical_column_indices)
            method = random.choice(['normalize', 'standardize', 'robust_scale'])
            slice_start = 0
            slice_end = 5 

            if func_name == 'transform_column':
                result = func(table_data, col, method)
                if result is not None:
                    full_result = result.tolist()
                    result_slice, slice_start, slice_end = get_random_slice(full_result)
                    params = {
                        'col': col,
                        'method': method,
                        'slice_start': slice_start,
                        'slice_end': slice_end,
                        'transformed_values': result_slice 
                    }
                else:
                    params = None

            else:  # stepwise_transform_column
                result = func(table_data, col, method)
                if result is not None:
                    # Get slice of transformed values
                    full_result = result['transformed_values'].tolist()
                    result_slice, slice_start, slice_end = get_random_slice(full_result)
                    
                    if method == 'normalize':
                        transformation_steps = f"""   - Minimum value: {result['min_val']:.3f}
                        - Maximum value: {result['max_val']:.3f}
                        - Formula: (x - min) / (max - min)"""

                        properties = """   - All values are scaled to range [0, 1]
                        - Preserves zero values
                        - Preserves distribution shape"""

                        transformed_stats = f"""   - Minimum: {result['transformed_stats']['min']:.3f} (should be 0)
                        - Maximum: {result['transformed_stats']['max']:.3f} (should be 1)
                        - Mean: {result['transformed_stats']['mean']:.3f}"""

                    elif method == 'standardize':
                        transformation_steps = f"""   - Mean: {result['mean']:.3f}
                        - Standard deviation: {result['std']:.3f}
                        - Formula: (x - mean) / std"""

                        properties = """   - Centered around zero
                        - Unit variance
                        - Preserves outliers
                        - Useful for normally distributed features"""

                        transformed_stats = f"""   - Mean: {result['transformed_stats']['mean']:.3f} (should be ~0)
                        - Standard deviation: {result['transformed_stats']['std']:.3f} (should be ~1)"""

                    elif method == 'robust_scale':
                        transformation_steps = f"""   - Median: {result['median']:.3f}
                        - Q1 (25th percentile): {result['q1']:.3f}
                        - Q3 (75th percentile): {result['q3']:.3f}
                        - IQR: {result['iqr']:.3f}
                        - Formula: (x - median) / IQR"""

                        properties = """   - Robust to outliers
                        - Based on percentiles
                        - Preserves zero values
                        - Useful for data with outliers"""

                        transformed_stats = f"""   - Median: {result['transformed_stats']['median']:.3f}
                        - IQR: {result['transformed_stats']['iqr']:.3f}"""

                    params = {
                        'col': col,
                        'method': method,
                        'original_stats': {k: round(v, 3) for k, v in result['original_stats'].items()},
                        'transformation_steps': transformation_steps,
                        'transformed_stats': transformed_stats,
                        'properties': properties,
                        'slice_start': slice_start,
                        'slice_end': slice_end,
                        'transformed_values': result_slice
                    }
                else:
                    params = None
        else:
            params = None
    
    elif func_name == 'log_transform_column':
        if numerical_column_indices:
            col = random.choice(numerical_column_indices)
            result = func(table_data, col)
            if result is not None:
                full_result = result.tolist()
                result_slice, slice_start, slice_end = get_random_slice(full_result)
                params = {
                    'col': col,
                    'transformed_values': result_slice,
                    'slice_start': slice_start,
                    'slice_end': slice_end
                }
            else:
                params = None
        else:
            params = None

    return params

def execute_causal_function(func_name, causal_info):
    if func_name == "infer_feature_causality":
        positive_causal_relationships = [causal_relationship for causal_relationship in causal_info if causal_relationship["is_causal"]]
        if positive_causal_relationships:
            selected_causal_relationship = random.choice(positive_causal_relationships)
        else:
            selected_causal_relationship = random.choice(causal_info)
        params = {
            "source_feature": selected_causal_relationship['source_feature'],
            "sink_feature": selected_causal_relationship['sink_feature'],
            "is_causal": "Yes" if selected_causal_relationship['is_causal'] else "No",
            "sentence_filler": "a" if selected_causal_relationship['is_causal'] else "no"
        }
    return params
