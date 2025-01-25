import numpy as np
import pandas as pd
import statistics

def find_value(df, row, col):
    """
    Find a value in a specific cell of the DataFrame.
    
    Args:
        df: pandas DataFrame to search in
        row: row index (integer)
        col: column index (integer)
    
    Returns cell value if coordinates are valid, None otherwise
    """
    if 0 <= row < df.shape[0] and 0 <= col < df.shape[1]:
        return df.iloc[row, col]
    return None

def compare_rows_or_columns(df, index1, index2, axis):
    """
    Compare two rows or columns for equality.
    
    Args:
        df: pandas DataFrame to analyze
        index1: first row/column index to compare
        index2: second row/column index to compare
        axis: 0 for rows, 1 for columns
    
    Returns True if the rows/columns are identical, False otherwise
    """
    if axis == 0:  # compare rows
        return df.iloc[index1].equals(df.iloc[index2])
    elif axis == 1:  # compare columns
        return df.iloc[:, index1].equals(df.iloc[:, index2])
    return False

def non_empty_cells(df, index, axis):
    """
    Find all non-empty (non-NaN) cells in a row or column.
    
    Args:
        df: pandas DataFrame to analyze
        index: row/column index to check
        axis: 0 for row, 1 for column
    
    Returns array of non-empty values
    """
    if axis == 0:  # row
        return df.iloc[index].dropna().values
    elif axis == 1:  # column
        return df.iloc[:, index].dropna().values

def unique_values(df, index, axis):
    """
    Find all unique values in a row or column.
    
    Args:
        df: pandas DataFrame to analyze
        index: row/column index to check
        axis: 0 for row, 1 for column
    
    Returns array of unique values
    """
    if axis == 0:  # row
        return df.iloc[index].unique()
    elif axis == 1:  # column
        return df.iloc[:, index].unique()

def value_present(df, index, value, axis):
    """
    Check if a specific value exists in a row or column.
    
    Args:
        df: pandas DataFrame to analyze
        index: row/column index to check
        value: value to search for
        axis: 0 for row, 1 for column
    
    Returns True if value is found, False otherwise
    """
    if axis == 0:  # row
        return value in df.iloc[index].values
    elif axis == 1:  # column
        return value in df.iloc[:, index].values

def fetch_neighbors(df, row, col):
    """
    Get the neighboring values of a cell (above, below, left, right).
    
    Args:
        df: pandas DataFrame to analyze
        row: row index of target cell
        col: column index of target cell
    
    Returns dictionary with neighboring values where they exist
    """
    neighbors = {}
    if row > 0:
        neighbors['above'] = df.iloc[row - 1, col]
    if row < df.shape[0] - 1:
        neighbors['below'] = df.iloc[row + 1, col]
    if col > 0:
        neighbors['left'] = df.iloc[row, col - 1]
    if col < df.shape[1] - 1:
        neighbors['right'] = df.iloc[row, col + 1]
    return neighbors

def top_n_values(df, index, n, axis):
    """
    Get the N largest values in a row or column.
    
    Args:
        df: pandas DataFrame to analyze
        index: row/column index to check
        n: number of top values to return
        axis: 0 for row, 1 for column
    
    Returns series of n largest values
    """
    if axis == 0:  # row
        return df.iloc[index].nlargest(n)
    elif axis == 1:  # column
        return df.iloc[:, index].nlargest(n)

def bottom_n_values(df, index, n, axis):
    """
    Get the N smallest values in a row or column.
    
    Args:
        df: pandas DataFrame to analyze
        index: row/column index to check
        n: number of bottom values to return
        axis: 0 for row, 1 for column
    
    Returns series of n smallest values
    """
    if axis == 0:  # row
        return df.iloc[index].nsmallest(n)
    elif axis == 1:  # column
        return df.iloc[:, index].nsmallest(n)

def sort_values(df, index, ascending, axis):
    """
    Sort values in a row or column.
    
    Args:
        df: pandas DataFrame to analyze
        index: row/column index to sort
        ascending: True for ascending order, False for descending
        axis: 0 for row, 1 for column
    
    Returns sorted series
    """
    if axis == 0:  # row
        return df.iloc[index].sort_values(ascending=ascending)
    elif axis == 1:  # column
        return df.iloc[:, index].sort_values(ascending=ascending)

def max_min_value(df, index, axis):
    """
    Find maximum and minimum values in a row or column.
    
    Args:
        df: pandas DataFrame to analyze
        index: row/column index to check
        axis: 0 for row, 1 for column
    
    Returns tuple of (max_value, min_value)
    """
    if axis == 0:  # row
        return df.iloc[index].max(), df.iloc[index].min()
    elif axis == 1:  # column
        return df.iloc[:, index].max(), df.iloc[:, index].min()

def add_cells(df, cell1, cell2):
    """
    Add values from two cells.
    
    Args:
        df: pandas DataFrame to analyze
        cell1: tuple of (row, col) for first cell
        cell2: tuple of (row, col) for second cell
    
    Returns sum of the two cell values
    """
    return df.iloc[cell1[0], cell1[1]] + df.iloc[cell2[0], cell2[1]]

def subtract_cells(df, cell1, cell2):
    """
    Subtract values of two cells.
    
    Args:
        df: pandas DataFrame to analyze
        cell1: tuple of (row, col) for first cell
        cell2: tuple of (row, col) for second cell
    
    Returns difference between the two cell values (cell1 - cell2)
    """
    return df.iloc[cell1[0], cell1[1]] - df.iloc[cell2[0], cell2[1]]

def sum_values(df, index, axis):
    """
    Calculate sum of values in a row or column.
    
    Args:
        df: pandas DataFrame to analyze
        index: row/column index to sum
        axis: 0 for row, 1 for column
    
    Returns sum of values
    """
    if axis == 0:  # row
        return df.iloc[index].sum()
    elif axis == 1:  # column
        return df.iloc[:, index].sum()

def add_constant(df, index, constant, axis):
    """
    Add a constant value to all elements in a row or column.
    
    Args:
        df: pandas DataFrame to analyze
        index: row/column index to modify
        constant: value to add
        axis: 0 for row, 1 for column
    
    Returns series with constant added to each value
    """
    if axis == 0:  # row
        return df.iloc[index] + constant
    elif axis == 1:  # column
        return df.iloc[:, index] + constant

def table_shape(df):
    """
    Get the dimensions of the DataFrame.
    
    Args:
        df: pandas DataFrame to analyze
    
    Returns tuple of (rows, columns)
    """
    return df.shape

def has_missing_values(df, index, axis):
    """
    Check if a row or column has any missing values.
    
    Args:
        df: pandas DataFrame to analyze
        index: row/column index to check
        axis: 0 for row, 1 for column
    
    Returns True if missing values exist, False otherwise
    """
    if axis == 0:  # row
        return df.iloc[index].isna().any()
    elif axis == 1:  # column
        return df.iloc[:, index].isna().any()

def count_empty_cells(df, index, axis):
    """
    Count number of missing values in a row or column.
    
    Args:
        df: pandas DataFrame to analyze
        index: row/column index to check
        axis: 0 for row, 1 for column
    
    Returns count of missing values
    """
    if axis == 0:  # row
        return df.iloc[index].isna().sum()
    elif axis == 1:  # column
        return df.iloc[:, index].isna().sum()

def fill_missing_values(df, index, value, axis):
    """
    Replace missing values in a row or column with a specified value.
    
    Args:
        df: pandas DataFrame to analyze
        index: row/column index to fill
        value: value to use for filling missing values
        axis: 0 for row, 1 for column
    
    Returns series with missing values filled
    """
    if axis == 0:  # row
        return df.iloc[index].fillna(value)
    elif axis == 1:  # column
        return df.iloc[:, index].fillna(value)

def find_all_cells_with_value(df, value):
    """
    Find all cells containing a specific value.
    
    Args:
        df: pandas DataFrame to analyze
        value: value to search for
    
    Returns list of (row, column) tuples where value is found
    """
    mask = df == value
    return list(zip(*np.where(mask)))

def count_value_occurrences(df, value):
    """
    Count how many times a value appears in the DataFrame.
    
    Args:
        df: pandas DataFrame to analyze
        value: value to count
    
    Returns total count of value occurrences
    """
    return (df == value).sum().sum()

def sum_table_values(df):
    """
    Calculate sum of all values in the DataFrame.
    
    Args:
        df: pandas DataFrame to analyze
    
    Returns sum of all numeric values
    """
    return df.sum().sum()

def average_table_values(df):
    """
    Calculate average of all values in the DataFrame.
    
    Args:
        df: pandas DataFrame to analyze
    
    Returns mean of all numeric values
    """
    return df.mean().mean()

def cumulative_sum(df, index, axis):
    """
    Calculate cumulative sum along a row or column.
    
    Args:
        df: pandas DataFrame to analyze
        index: row/column index to process
        axis: 0 for row, 1 for column
    
    Returns series of cumulative sums
    """
    if axis == 0:  # row
        return df.iloc[index].cumsum()
    elif axis == 1:  # column
        return df.iloc[:, index].cumsum()

def successive_differences(df, index, axis):
    """
    Calculate differences between successive elements in a row or column.
    
    Args:
        df: pandas DataFrame to analyze
        index: row/column index to process
        axis: 0 for row, 1 for column
    
    Returns series of differences between consecutive values
    """
    if axis == 0:  # row
        return df.iloc[index].diff()
    elif axis == 1:  # column
        return df.iloc[:, index].diff()

def value_range(df, index, axis):
    """
    Calculate the range (max - min) of values in a row or column.
    
    Args:
        df: pandas DataFrame to analyze
        index: row/column index to process
        axis: 0 for row, 1 for column
    
    Returns difference between maximum and minimum values
    """
    if axis == 0:  # row
        return df.iloc[index].max() - df.iloc[index].min()
    elif axis == 1:  # column
        return df.iloc[:, index].max() - df.iloc[:, index].min()

def mean_value(df, index, axis):
    """
    Calculate mean value of a row or column.
    
    Args:
        df: pandas DataFrame to analyze
        index: row/column index to process
        axis: 0 for row, 1 for column
    
    Returns mean value
    """
    if axis == 0:  # row
        return df.iloc[index].mean()
    elif axis == 1:  # column
        return df.iloc[:, index].mean()

def median_value(df, index, axis):
    """
    Calculate median value of a row or column.
    
    Args:
        df: pandas DataFrame to analyze
        index: row/column index to process
        axis: 0 for row, 1 for column
    
    Returns median value
    """
    if axis == 0:  # row
        return df.iloc[index].median()
    elif axis == 1:  # column
        return df.iloc[:, index].median()

def mode_value(df, index, axis):
    """
    Calculate mode (most frequent value) of a row or column.
    
    Args:
        df: pandas DataFrame to analyze
        index: row/column index to process
        axis: 0 for row, 1 for column
    
    Returns mode value
    """
    if axis == 0:  # row
        return df.iloc[index].mode().iloc[0] if not df.iloc[index].mode().empty else None
    elif axis == 1:  # column
        return df.iloc[:, index].mode().iloc[0] if not df.iloc[:, index].mode().empty else None

def square_values(df, index, axis):
    """
    Square all values in a row or column.
    
    Args:
        df: pandas DataFrame to analyze
        index: row/column index to process
        axis: 0 for row, 1 for column
    
    Returns series with squared values
    """
    if axis == 0:  # row
        return df.iloc[index] ** 2
    elif axis == 1:  # column
        return df.iloc[:, index] ** 2

def sqrt_values(df, index, axis):
    """
    Calculate square root of all values in a row or column.
    Only processes non-negative values.
    
    Args:
        df: pandas DataFrame to analyze
        index: row/column index to process
        axis: 0 for row, 1 for column
    
    Returns series with square root of non-negative values
    """
    if axis == 0:  # row
        return df.iloc[index][df.iloc[index] >= 0].apply(np.sqrt)
    elif axis == 1:  # column
        return df.iloc[:, index][df.iloc[:, index] >= 0].apply(np.sqrt)

def abs_values(df, index, axis):
    """
    Calculate absolute values of all elements in a row or column.
    
    Args:
        df: pandas DataFrame to analyze
        index: row/column index to process
        axis: 0 for row, 1 for column
    
    Returns series with absolute values
    """
    if axis == 0:  # row
        return df.iloc[index].abs()
    elif axis == 1:  # column
        return df.iloc[:, index].abs()

def direct_correlation_between_columns(df, col1, col2):
    """
    Calculate the Pearson correlation coefficient between two columns.
    
    Args:
        df: pandas DataFrame to analyze
        col1: first column index
        col2: second column index
    
    Returns correlation coefficient between -1 and 1, or None if calculation is not possible
    """
    try:
        return df.iloc[:, col1].corr(df.iloc[:, col2])
    except:
        return None

def stepwise_correlation_between_columns(df, col1, col2):
    """
    Calculate the Pearson correlation coefficient between two columns
    and provide stepwise calculation details.
    
    Args:
        df: pandas DataFrame to analyze
        col1: first column index
        col2: second column index
    
    Returns dictionary containing correlation coefficient and intermediate steps
    """
    try:
        x = df.iloc[:, col1].dropna()
        y = df.iloc[:, col2].dropna()
        
        valid_indices = x.index.intersection(y.index)
        x = x[valid_indices]
        y = y[valid_indices]
        
        x_mean = x.mean()
        y_mean = y.mean()

        x_dev = x - x_mean
        y_dev = y - y_mean
        
        sum_product_dev = (x_dev * y_dev).sum()
        
        sum_squared_dev_x = (x_dev ** 2).sum()
        sum_squared_dev_y = (y_dev ** 2).sum()
        
        correlation = sum_product_dev / (sum_squared_dev_x * sum_squared_dev_y) ** 0.5
        
        return {
            'correlation': correlation,
            'x_mean': x_mean,
            'y_mean': y_mean,
            'n_samples': len(x),
            'sum_product_dev': sum_product_dev,
            'sum_squared_dev_x': sum_squared_dev_x,
            'sum_squared_dev_y': sum_squared_dev_y
        }
    except:
        return None

def transform_column(df, index, method):
    """
    Apply direct transformation to a column.
    
    Args:
        df: pandas DataFrame to analyze
        index: column index to transform
        method: transformation method ('normalize', 'standardize', 'robust_scale', 'log')
    
    Returns transformed series or None if transformation is not possible
    """
    try:
        series = df.iloc[:, index].dropna()
        
        if method == 'normalize':  # Min-Max Normalization
            return (series - series.min()) / (series.max() - series.min())
            
        elif method == 'standardize':  # Z-score Standardization
            return (series - series.mean()) / series.std()
            
        elif method == 'robust_scale':  # Robust Scaling using IQR
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            return (series - series.median()) / IQR
            
    except:
        return None

def stepwise_transform_column(df, index, method):
    """
    Apply transformation to a column with detailed steps.
    
    Args:
        df: pandas DataFrame to analyze
        index: column index to transform
        method: transformation method ('normalize', 'standardize', 'robust_scale', 'log')
    
    Returns dictionary containing transformed values and intermediate steps
    """
    try:
        series = df.iloc[:, index].dropna()
        original_stats = {
            'mean': series.mean(),
            'std': series.std(),
            'min': series.min(),
            'max': series.max(),
            'median': series.median(),
            'q1': series.quantile(0.25),
            'q3': series.quantile(0.75)
        }
        
        if method == 'normalize':
            min_val = series.min()
            max_val = series.max()
            transformed = (series - min_val) / (max_val - min_val)
            
            return {
                'method': 'normalize',
                'original_stats': original_stats,
                'min_val': min_val,
                'max_val': max_val,
                'transformed_values': transformed,
                'transformed_stats': {
                    'min': transformed.min(),  # Should be 0
                    'max': transformed.max(),  # Should be 1
                    'mean': transformed.mean()
                }
            }
            
        elif method == 'standardize':
            mean_val = series.mean()
            std_val = series.std()
            transformed = (series - mean_val) / std_val
            
            return {
                'method': 'standardize',
                'original_stats': original_stats,
                'mean': mean_val,
                'std': std_val,
                'transformed_values': transformed,
                'transformed_stats': {
                    'mean': transformed.mean(),  # Should be ~0
                    'std': transformed.std()     # Should be ~1
                }
            }
            
        elif method == 'robust_scale':
            median_val = series.median()
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            transformed = (series - median_val) / iqr
            
            return {
                'method': 'robust_scale',
                'original_stats': original_stats,
                'median': median_val,
                'q1': q1,
                'q3': q3,
                'iqr': iqr,
                'transformed_values': transformed,
                'transformed_stats': {
                    'median': transformed.median(),
                    'iqr': transformed.quantile(0.75) - transformed.quantile(0.25)
                }
            }
            
    except:
        return None

def log_transform_column(df, index):
    """
    Apply log transformation to a column.
    Takes absolute value before log transform to handle negative values.
    
    Args:
        df: pandas DataFrame to analyze
        index: column index to transform
    
    Returns log-transformed series or None if transformation is not possible
    """
    try:
        series = df.iloc[:, index].dropna()
        series = np.abs(series) + 1e-10
        return np.log(series)
    except:
        return None

def infer_feature_causality():
    return None