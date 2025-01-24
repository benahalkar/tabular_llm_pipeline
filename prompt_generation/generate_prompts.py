import random
import json
import importlib
import inspect
from pathlib import Path
import pandas as pd
from function_executor import execute_function, execute_causal_function
from prompt_templates import initialize_prompt_templates
import yaml


class TablePromptGenerator:
    def __init__(self, metadata_file, functions_module, processed_data_path):
        """
        Initialize generator with metadata file path and functions module name.
        """
        self.metadata = self.load_metadata(metadata_file)
        self.functions = self.load_functions(functions_module)
        self.prompt_templates = initialize_prompt_templates()
        self.processed_data_path = processed_data_path

    def load_metadata(self, filename):
        """Load table metadata from JSON file."""
        with open(filename, 'r') as f:
            return json.load(f)

    def load_functions(self, module_name):
        """
        Dynamically load all eligible functions from the specified module.
        Returns a dictionary mapping function names to function objects.
        """
        module = importlib.import_module(module_name)
        functions = {}
        
        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj):
                functions[name] = obj
        return functions

    def load_table(self, filename, is_synthetic_data):
        """Load CSV table data."""
        if is_synthetic_data:
            return pd.read_csv(filename, header=None)
        else: 
            return pd.read_csv(filename, header=0)

    def generate_prompt(self, func_name, table_name, params):
        """Generate human and GPT prompts based on function and parameters."""
        template = self.get_template_for_function(func_name)
        question = template['question'].format(table=table_name, **params)
        answer = template['answer'].format(table=table_name, **params)
        
        return {
            "from": "human",
            "value": question
        }, {
            "from": "gpt",
            "value": answer
        }

    def get_template_for_function(self, func_name):
        """Get the appropriate template for a function based on its name."""
        for template_name, template in self.prompt_templates.items():
            if template_name == func_name:
                return template
        raise ValueError(f"No template found for function: {func_name}")

    def generate_random_prompt(self, prompt_id, table_encoder_type):
        """Generate a single random prompt."""
        selected_table_metadata = random.choice(self.metadata)
        selected_table = selected_table_metadata['table_name']
        print(selected_table)
        if selected_table.startswith("Causal_Table") and selected_table_metadata['has_causal'] is True:
            selected_func = 'infer_feature_causality'
            params = execute_causal_function(selected_func, selected_table_metadata['causal_info'])
        else:
            acceptable_functions = list(self.functions.keys())
            acceptable_functions.remove('infer_feature_causality')
            selected_func = random.choice(acceptable_functions)
            print(selected_func)
            num_cols = selected_table_metadata['num_cols']
            num_rows = selected_table_metadata['num_rows']
            is_synthetic_data = selected_table_metadata['is_synthetic_data']
            categorical_columns = selected_table_metadata['categorical_columns']
            label_index = selected_table_metadata['label_index']
            table_data = self.load_table(self.processed_data_path + selected_table, is_synthetic_data)
            params = execute_function(selected_func, table_data, self.functions, categorical_columns, label_index, table_encoder_type, num_cols-1, num_rows)
        if params is None:
            print(f"Skipping Prompt {prompt_id}")
            return None
        human_msg, gpt_msg = self.generate_prompt(selected_func, selected_table, params)

        print(f"Generated Prompt {prompt_id}")
        
        return {
            "id": str(prompt_id),
            "table": f"{selected_table}",
            "conversations": [human_msg, gpt_msg]
        }

    def generate_dataset(self, num_prompts, table_encoder_type):
        """Generate multiple prompts."""
        prompt_list = []
        for i in range(num_prompts):
            prompt = self.generate_random_prompt(i, table_encoder_type)
            if prompt is None:
                continue
            else:
                prompt_list.append(prompt)
        return prompt_list


def main():
    config_file_path = 'config/preprocess_config.yaml'
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)
    
    generator = TablePromptGenerator(config['data_settings']['processed_metadata_file'], config['prompt_settings']['functions_file'], 
                                     config['data_settings']['processed_data_path'])
    
    # Generate prompts
    num_prompts = config['prompt_settings']['num_prompts']
    table_encoder_type = config['data_settings']['table_encoder_type']
    is_target_masked = config['data_settings']['apply_target_mask']
    mask_token = config['data_settings']['mask_token']

    prompts = generator.generate_dataset(num_prompts, table_encoder_type)

    prompts_file_path = config['prompt_settings']['prompts_file_path']
    with open(prompts_file_path, 'w') as f:
        json.dump(prompts, f, indent=2)


if __name__ == "__main__":
    main()
