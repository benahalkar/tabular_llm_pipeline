def initialize_prompt_templates():
    """Initialize templates for different types of functions."""
    return {
        'find_value': {
            'question': "Find the value in the given table, in row {row} column {col}\n<table>",
            'answer': "The value at Row {row} and Column {col} is {result}"
        },
        'compare_rows_or_columns': {
            'question': "Are {axis_name} {index1} and {axis_name} {index2} identical?\n<table>",
            'answer': "The {axis_name}s are{presence} identical"
        },
        'non_empty_cells': {
            'question': "What are the non-empty values in {axis_name} {index} present from index {slice_start} to index {slice_end}?\n<table>",
            'answer': "The non-empty values in {axis_name} {index} from index {slice_start} to {slice_end} are: {result}"
        },
        'unique_values': {
            'question': "What are the unique values in {axis_name} {index} (showing slice from index {slice_start} to {slice_end})?\n<table>",
            'answer': "The unique values in {axis_name} {index} from index {slice_start} to {slice_end} are: {result}"
        },
        'value_present': {
            'question': "Is the value {value} present in {axis_name} {index}?\n<table>",
            'answer': "The value {value} is{presence} present in {axis_name} {index}"
        },
        'fetch_neighbors': {
            'question': "What are the neighboring values for the cell at row {row} and column {col}?\n<table>",
            'answer': "The neighboring values are: {result}"
        },
        'top_n_values': {
            'question': "What are the top {n} values in {axis_name} {index}?\n<table>",
            'answer': "The top {n} values in {axis_name} {index} are: {result}"
        },
        'bottom_n_values': {
            'question': "What are the bottom {n} values in {axis_name} {index}?\n<table>",
            'answer': "The bottom {n} values in {axis_name} {index} are: {result}"
        },
        'sort_values': {
            'question': "What are the values in {axis_name} {index} sorted in {order} order? Return the slice from index {slice_start} to {slice_end}.\n<table>",
            'answer': "The sorted values in {axis_name} {index} from index {slice_start} to {slice_end} are: {result}"
        },
        'max_min_value': {
            'question': "What are the maximum and minimum values in {axis_name} {index}?\n<table>",
            'answer': "The maximum value is {max_val} and the minimum value is {min_val}"
        },
        'add_cells': {
            'question': "What is the sum of values at positions {cell1} and {cell2}?\n<table>",
            'answer': "The sum of the values is {result}"
        },
        'subtract_cells': {
            'question': "What is the difference between values at positions {cell1} and {cell2}?\n<table>",
            'answer': "The difference between the values is {result}"
        },
        'sum_values': {
            'question': "What is the sum of all values in {axis_name} {index}?\n<table>",
            'answer': "The sum of values in {axis_name} {index} is {result}"
        },
        'add_constant': {
            'question': "What are the values after adding {constant} to {axis_name} {index} present from the index {slice_start} to index {slice_end}?\n<table>",
            'answer': "The values after adding {constant} in {axis_name} {index} from index {slice_start} to {slice_end} are: {result}"
        },
        'table_shape': {
            'question': "What are the dimensions of the table?\n<table>",
            'answer': "The table has {rows} rows and {cols} columns"
        },
        'has_missing_values': {
            'question': "Are there any missing values in {axis_name} {index}?\n<table>",
            'answer': "There {presence} missing values in {axis_name} {index}"
        },
        'count_empty_cells': {
            'question': "How many empty cells are in {axis_name} {index}?\n<table>",
            'answer': "There are {result} empty cells in {axis_name} {index}"
        },
        'fill_missing_values': {
            'question': "What are the values after filling missing values with {value} in {axis_name} {index} present from the index {slice_start} to index {slice_end}?\n<table>",
            'answer': "The values after filling missing values with {value} in {axis_name} {index} present from the index {slice_start} to index {slice_end} are: {result}"
        },
        'find_all_cells_with_value': {
            'question': "Where does the value {value} appear in the table?\n<table>",
            'answer': "The value {value} appears at positions: {result}"
        },
        'count_value_occurrences': {
            'question': "How many times does the value {value} appear in the table?\n<table>",
            'answer': "The value {value} appears {result} times"
        },
        'sum_table_values': {
            'question': "What is the sum of all values in the table?\n<table>",
            'answer': "The sum of all values is {result}"
        },
        'average_table_values': {
            'question': "What is the average of all values in the table?\n<table>",
            'answer': "The average of all values is {result}"
        },
        'cumulative_sum': {
            'question': "What is the cumulative sum of values in {axis_name} {index}? Only show the slice from index {slice_start} to {slice_end}.\n<table>",
            'answer': "The cumulative sum in {axis_name} {index} from index {slice_start} to {slice_end} is: {result}"
        },
        'successive_differences': {
            'question': "What are the differences between successive values in {axis_name} {index} (showing slice from index {slice_start} to {slice_end})?\n<table>",
            'answer': "The successive differences in {axis_name} {index} from index {slice_start} to {slice_end} are: {result}"
        },
        'value_range': {
            'question': "What is the range of values in {axis_name} {index}?\n<table>",
            'answer': "The range of values is {result}"
        },
        'mean_value': {
            'question': "What is the mean value of {axis_name} {index}?\n<table>",
            'answer': "The mean value is {result}"
        },
        'median_value': {
            'question': "What is the median value of {axis_name} {index}?\n<table>",
            'answer': "The median value is {result}"
        },
        'mode_value': {
            'question': "What is the most frequent value in {axis_name} {index}?\n<table>",
            'answer': "The most frequent value is {result}"
        },
        'square_values': {
            'question': "What are the squared values in {axis_name} {index}? Return the slice from index {slice_start} to {slice_end}.\n<table>",
            'answer': "The squared values in {axis_name} {index} from index {slice_start} to {slice_end} are: {result}"
        },
        'sqrt_values': {
            'question': "What are the square roots of non negative values in {axis_name} {index} (showing slice from index {slice_start} to {slice_end})?\n<table>",
            'answer': "The square roots in {axis_name} {index} from index {slice_start} to {slice_end} are: {result}"
        },
        'abs_values': {
            'question': "What are the absolute values in {axis_name} {index} present in the slice from index {slice_start} to index {slice_end}?\n<table>",
            'answer': "The absolute values in {axis_name} {index} from index {slice_start} to {slice_end} are: {result}"
        },
        'direct_correlation_between_columns': {
            'question': "What is the pearson correlation coefficient between column {col1} and column {col2} in the table?\n<table>",
            'answer': "The pearson correlation coefficient between column {col1} and column {col2} in the table is {correlation}."
        },
        'stepwise_correlation_between_columns': {
            'question': "In a stepwise fashion, calculate and explain the correlation coefficient between column {col1} and column {col2} of the table.\n<table>",
            'answer': """Let me explain the correlation calculation between column {col1} and column {col2} of the given table in a step by step manner:

            1. First, I analyzed all {n_samples} valid data points by excluding any missing values.

            2. I calculated the means:
            - Mean of column {col1}: {x_mean}
            - Mean of column {col2}: {y_mean}

            3. Then, I calculated:
            - Sum of products of deviations: {sum_product_dev}
            - Sum of squared deviations for column {col1}: {sum_squared_dev_x}
            - Sum of squared deviations for column {col2}: {sum_squared_dev_y}

            4. Finally, I computed the correlation coefficient using the formula:
            correlation = sum_product_dev / sqrt(sum_squared_dev_x * sum_squared_dev_y)

            The final correlation coefficient is {correlation}.

            This means that the two columns have a correlation of {correlation}."""
        },
        'transform_column': {
            'question': "Transform column {col} of the table using the {method} transformation. Return the slice from index {slice_start} to {slice_end}.\n<table>",
            'answer': "Here's a sample of the transformed values of column {col} from the table, using the {method} transformation (indices {slice_start} to {slice_end}): {transformed_values}."
        },
        'stepwise_transform_column': {
            'question': "Explain the {method} transformation process for column {col} for the given table. Make sure you return values from index {slice_start} to index {slice_end}.\n<table>",
            'answer': """Let me explain the {method} transformation process for column {col} of the given table in a step by step manner:

            1. Original Data Statistics:
                - Mean: {original_stats[mean]:.3f}
                - Standard Deviation: {original_stats[std]:.3f}
                - Min: {original_stats[min]:.3f}
                - Max: {original_stats[max]:.3f}
                - Median: {original_stats[median]:.3f}

            2. Transformation Process:
            {transformation_steps}

            3. Transformed Data Statistics:
            {transformed_stats}

            4. Key Properties of this Transformation:
            {properties}

            5. Sample of transformed values (indices {slice_start} to {slice_end}):
            {transformed_values}"""
        },
        'log_transform_column': {
            'question': "Apply the log transformation to the absolute values of column {col} of the table. Return the slice from index {slice_start} to {slice_end}.\n<table>",
            'answer': "Here's a sample of the log-transformed values of the absolute values for column {col} from the table, from index {slice_start} to index {slice_end}: {transformed_values}"
        },
        'infer_feature_causality': {
            'question': "In the given table, is there any causal relationship between features {source_feature} and {sink_feature}? Meaning, does feature {source_feature} cause feature {sink_feature}?\n<table>",
            'answer': "{is_causal}, there is {sentence_filler} causal relationship between feature {source_feature} and feature {sink_feature}."
        }
    }