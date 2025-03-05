import os
import sys
import time
import json
import pytz
import torch
import wandb
import warnings
import numpy as np
import pandas as pd
from copy import deepcopy
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Optional, List

from accelerate import Accelerator
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    HfArgumentParser,
    TrainerCallback,
    PreTrainedTokenizer,
    set_seed as transformers_set_seed,
)
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown

import conversation as conversation_lib
from constants import (
    IGNORE_INDEX,
    TABLE_TOKEN_INDEX,
    DEFAULT_TABLE_TOKEN,
    DEFAULT_TABLE_START_TOKEN,
    DEFAULT_TABLE_END_TOKEN,
    TABLE_PLACEHOLDER,
    COL_SEPARATOR,
    ROW_SEPARATOR,
)

# Initialize accelerator
accelerator = Accelerator()
local_rank = None  # Define local_rank globally

@dataclass
class CustomModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: Optional[str] = field(
        default="facebook/opt-125m",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    attn_implementation: str = field(
        default=None,
        metadata={"help": "Attention implementation to use."},
    )

@dataclass
class CustomDataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    data_path: str = field(
        default=None, 
        metadata={"help": "Path to the training data."}
    )

@dataclass
class CustomTrainingArguments(TrainingArguments):
    """
    Arguments pertaining to training parameters.
    """
    device: str = field(
        default="cpu", 
        metadata={"help": "Device to use."}
    )
    cache_dir: Optional[str] = field(
        default=None, 
        metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    optim: str = field(
        default="adamw_torch", 
        metadata={"help": "Optimizer to use."}
    )
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."},
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."},
    )
    bits: int = field(
        default=16, 
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = field(
        default=False, 
        metadata={"help": "Enable LoRA training."}
    )
    lora_r: int = field(
        default=64, 
        metadata={"help": "LoRA attention dimension."}
    )
    lora_threshold: float = field(
        default=6.0, 
        metadata={"help": "LoRA attention threshold."}
    )
    lora_alpha: int = field(
        default=16, 
        metadata={"help": "LoRA scaling parameter."}
    )
    lora_dropout: float = field(
        default=0.05, 
        metadata={"help": "LoRA dropout probability."}
    )
    lora_weight_path: str = field(
        default="", 
        metadata={"help": "LoRA weight path."}
    )
    lora_bias: str = field(
        default="none", 
        metadata={"help": "LoRA bias type."}
    )
    report_to: str = field(
        default=None, 
        metadata={"help": "Report to platform."}
    )
    distributed_state: str = field(
        default=None, 
        metadata={"help": "Distributed state."}
    )
    deepspeed: str = field(
        default=None, 
        metadata={"help": "Deepspeed configuration."}
    )

# Define the home directory
HOME_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize Weights & Biases if on the main process
if accelerator.is_main_process:
    wandb.init(project="finetuning_runs")

def rank0_print(msg: str) -> None:
    """
    Prints a message only on the rank 0 process.

    Args:
        msg (str): The message to print.
    """
    if local_rank == 0:
        print(msg)

def rank0_declare(msg: str) -> None:
    """
    Prints a declarative message only on the rank 0 process.

    Args:
        msg (str): The message to print.
    """
    if local_rank == 0:
        print("")
        print("#"*40)
        print(" "*10, msg.upper())
        print("#"*40)
        print("")

def get_timestamp() -> str:
    """
    Generates the current timestamp for logging.

    Returns:
        str: The formatted timestamp.
    """
    est_timezone = pytz.timezone('US/Eastern')
    current_est_time = datetime.now(est_timezone).strftime("%H:%M:%S")
    return f"Timestamp logged is {current_est_time}"

def get_filename() -> str:
    """
    Generates a unique filename for logging based on the current timestamp.

    Returns:
        str: The filepath for the log file.
    """
    est_timezone = pytz.timezone('US/Eastern')
    current_est_time = datetime.now(est_timezone).strftime("%Y%m%d_%H%M%S")
    log_directory = os.path.join(HOME_DIR, "training_logs")

    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    return os.path.join(log_directory, f"logs_{current_est_time}.txt")

# Define the log filename
FILENAME = get_filename()

def set_seed(seed: int = 42) -> None:
    """
    Sets the seed for reproducibility in torch, cuda, and transformers.

    Args:
        seed (int): The seed value. Defaults to 42.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers_set_seed(seed)
    torch.set_printoptions(threshold=float('inf'))
    rank0_print(f"Seed set to: {seed}")

def print_name_and_size(model: torch.nn.Module) -> None:
    """
    Prints the name and gradient requirement for each parameter in the model.

    Args:
        model (torch.nn.Module): The PyTorch model.
    """
    for name, param in model.named_parameters():
        rank0_print(f"Layer: {name} | Gradient: {param.requires_grad}")

def find_all_linear_names(model: torch.nn.Module) -> List[str]:
    """
    Finds the names of all linear layers in the model.

    Args:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        List[str]: A list of linear layer names.
    """
    cls = torch.nn.Linear
    lora_module_names = set()

    # Keywords for identifying multimodal modules (currently empty)
    # TODO: Complete this
    multimodal_keywords = []

    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    # Needed for 16-bit training
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')

    return list(lora_module_names)

def print_gpu_memory(message: str) -> None:
    """
    Prints GPU memory allocation information to the log file.

    Args:
        message (str): A message to prepend to the log entry.
    """
    if torch.cuda.is_available():
        with open(FILENAME, "a") as f:
            f.write("\n" + message + "\n")
            f.write(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB\n")
            f.write(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB\n")
        f.close()

def just_logging(message: str) -> None:
    """
    Logs a message to the log file.

    Args:
        message (str): The message to log.
    """
    if local_rank == 0:
        with open(FILENAME, "a") as f:
            timestamp = get_timestamp()
            f.write(timestamp, "\n")
            f.write('-'*len(timestamp), "\n")
            f.write(message + "\n")
            f.write("\n")
        f.close()

def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """
    Counts the total and trainable parameters in a model.

    Args:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        tuple[int, int]: A tuple containing the total and trainable parameter counts.
    """
    # total_params = sum(p.numel() for p in model.parameters())
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # return total_params, trainable_params

    total_params, trainable_params = 0, 0
    
    for name, param in model.named_parameters():
        if param is not None:
            num_params = param.numel()
            if 'lora' in name or any(peft_module in name for peft_module in ['adapter', 'prefix', 'prompt']):
                trainable_params += num_params
            if hasattr(param, 'quant_state'):
                # num_params = param.quant_state.num_elements()
                num_params = param.shape.numel()
            total_params += num_params
    
    return total_params, trainable_params

def analyze_model_datatypes(model: torch.nn.Module) -> Dict[str, int]:
    """
    Analyzes the datatypes of parameters in a PyTorch model.

    This function iterates through all parameters of the given model and counts
    the number of elements (parameters) for each datatype encountered.

    Args:
        model (torch.nn.Module): The PyTorch model to analyze.

    Returns:
        Dict[str, int]: A dictionary where keys are datatype strings (e.g., 'torch.float32')
        and values are the total number of parameters of that datatype.
    """
    datatype_counts = {}
    
    for param in model.parameters():
        dtype = str(param.dtype)
        num_params = param.numel()
        
        if dtype not in datatype_counts:
            datatype_counts[dtype] = 0
        datatype_counts[dtype] += num_params

    return datatype_counts


def estimate_model_size(model: torch.nn.Module) -> tuple[float, float]:
    """
    Estimates the size of a PyTorch model in MB and GB.

    Args:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        tuple[float, float]: A tuple containing the model size in MB and GB.
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    total_size_bytes = param_size + buffer_size
    total_size_mb = total_size_bytes / (1024**2)
    total_size_gb = total_size_bytes / (1024**3)

    return total_size_mb, total_size_gb

class MemoryMonitorCallback(TrainerCallback):
    """
    A TrainerCallback that monitors and logs GPU memory usage during training.
    """
    def __init__(self, print_interval: int = 1):
        """
        Initializes the MemoryMonitorCallback.

        Args:
            print_interval (int): The interval (in steps) at which to print memory usage.
        """
        self.print_interval = print_interval
        nvmlInit()
        self.handle = nvmlDeviceGetHandleByIndex(0)

    def on_step_end(self, args, state, control, **kwargs):
        """
        Called at the end of each training step.
        Logs GPU memory usage if the current step is a multiple of the print interval.
        """
        if state.global_step % self.print_interval == 0:
            info = nvmlDeviceGetMemoryInfo(self.handle)
            just_logging(f"Step {state.global_step}: GPU memory occupied: {info.used//1024**2} MB")

    def on_train_end(self, args, state, control, **kwargs):
        """
        Called at the end of training.
        Shuts down the NVML library.
        """
        nvmlShutdown()

def preprocess_prompts(conversations: List[Dict]) -> str:
    """
    Preprocesses conversation prompts to format them for the model.

    Args:
        conversations (List[Dict]): A list of dictionaries representing the conversation.

    Returns:
        str: The preprocessed prompt.
    """
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Ensure the conversation starts with a human and ends with a GPT response
    if roles[conversations[0]["from"]] != conv.roles[0]:
        conversations = conversations[1:]
    if roles[conversations[-1]["from"]] != conv.roles[1]:
        conversations = conversations[:-1]

    conv.messages = []
    for i, sentence in enumerate(conversations):
        role = roles[sentence["from"]]
        assert role == conv.roles[i % 2], "roles dont match"
        conv.append_message(role, sentence["value"])

    return conv.get_prompt()

def preprocess_jamba(conversations: str, tokenizer: AutoTokenizer) -> Dict[str, torch.Tensor]:
    """
    Preprocesses conversations for the Jamba model, tokenizing and creating labels with ignored indices.

    Args:
        conversations (str): The conversation string.
        tokenizer (AutoTokenizer): The tokenizer for the Jamba model.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing input IDs and labels.
    """
    def find_indexes(main_, sub_):
        main_array = np.array(main_)
        sub_array = np.array(sub_)
        window_view = np.lib.stride_tricks.sliding_window_view(main_array, len(sub_array))
        matches = np.all(window_view == sub_array, axis=1)
        return np.where(matches)[0]

    conv = tokenizer(conversations).input_ids
    assistant_command = "\nAssistant:"
    human_command = "\nHuman:"

    human_token = tokenizer.encode(human_command, add_special_tokens=False)[1:]
    human_indices = find_indexes(conv, human_token)

    assistant_token = tokenizer.encode(assistant_command, add_special_tokens=False)[1:]
    assistant_indices = find_indexes(conv, assistant_token) + len(assistant_token)

    assert len(human_indices) == len(assistant_indices), "Conversation is not identical"
    assert len(human_indices) > 0, "No indices found"

    input_ids = torch.tensor(conv)
    labels = input_ids.clone()

    first_index = assistant_indices[0]
    labels[:first_index] = IGNORE_INDEX

    human_indices, assistant_indices = human_indices[1:], assistant_indices[1:]
    for i, j in zip(human_indices, assistant_indices):
        labels[i:j] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=labels)

def preprocess_llama_3(conversations: str, tokenizer: AutoTokenizer) -> Dict[str, torch.Tensor]:
    """
    Preprocesses conversations for the Llama 3 model, tokenizing and creating labels with ignored indices.

    Args:
        conversations (str): The conversation string.
        tokenizer (AutoTokenizer): The tokenizer for the Llama 3 model.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing input IDs and labels.
    """
    def find_indexes(main_, sub_):
        main_array = np.array(main_)
        sub_array = np.array(sub_)
        window_view = np.lib.stride_tricks.sliding_window_view(main_array, len(sub_array))
        matches = np.all(window_view == sub_array, axis=1)
        return np.where(matches)[0]

    conv = tokenizer(conversations).input_ids
    assistant_command = "<|start_header_id|>assistant<|end_header_id|>"
    assistant_token = tokenizer.encode(assistant_command, add_special_tokens=False)
    assistant_indices = find_indexes(conv, assistant_token) + len(assistant_token)

    ending_command = "<|eot_id|>"
    ending_token = tokenizer.encode(ending_command, add_special_tokens=False)

    input_ids = torch.tensor(conv)
    labels = torch.full(input_ids.shape, fill_value=IGNORE_INDEX)

    for i, start in enumerate(assistant_indices):
        temp_mask = (input_ids == ending_token[0])
        temp_mask[:start] = False
        till = temp_mask.nonzero(as_tuple=True)[0][0].item()
        labels[start:till] = input_ids[start:till]

    return dict(input_ids=input_ids, labels=labels)

def preprocess_llama_2(conversations: str, tokenizer: AutoTokenizer) -> Dict[str, torch.Tensor]:
    """
    Preprocesses conversations for the Llama 2 model, tokenizing and creating labels with ignored indices.

    Args:
        conversations (str): The conversation string.
        tokenizer (AutoTokenizer): The tokenizer for the Llama 2 model.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing input IDs and labels.
    """
    def find_indexes(main_, sub_):
        main_array = np.array(main_)
        sub_array = np.array(sub_)
        window_view = np.lib.stride_tricks.sliding_window_view(main_array, len(sub_array))
        matches = np.all(window_view == sub_array, axis=1)
        return np.where(matches)[0]

    conv = tokenizer(conversations).input_ids
    user_command = "[INST] "
    assistant_command = " </s>"

    user_token = tokenizer.encode(user_command, add_special_tokens=False)
    user_indices = find_indexes(conv, user_token)

    assistant_token = tokenizer.encode(assistant_command, add_special_tokens=False)
    assistant_indices = find_indexes(conv, assistant_token) + len(assistant_token)

    assert len(user_indices) == len(assistant_indices), "Conversation is not identical"
    assert len(user_indices) > 0, "No indices found"

    input_ids = torch.tensor(conv)
    labels = input_ids.clone()

    first_index = assistant_indices[0]
    labels[:first_index] = IGNORE_INDEX

    user_indices, assistant_indices = user_indices[1:], assistant_indices[1:]
    for i, j in zip(user_indices, assistant_indices):
        labels[i:j] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=labels)

def preprocess(conversations: str, tokenizer: AutoTokenizer) -> Dict[str, torch.Tensor]:
    """
    Preprocesses conversations based on the conversation style.

    Args:
        conversations (str): The conversation string.
        tokenizer (AutoTokenizer): The tokenizer.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing input IDs and labels.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(conversations, tokenizer)
    elif conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_3:
        return preprocess_llama_3(conversations, tokenizer)
    elif conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.SINGLE:
        return preprocess_jamba(conversations, tokenizer)
    else:
        raise ValueError("Unknown conversation type found")

@dataclass
class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_args: CustomDataArguments, tokenizer: AutoTokenizer, train: bool = True):
        """
        Initializes the SupervisedDataset.

        Args:
            data_args (CustomDataArguments): Data-related arguments.
            tokenizer (AutoTokenizer): The tokenizer.
            train (bool): Whether this is a training dataset.
        """
        super(SupervisedDataset, self).__init__()
        self.dataset_folder = os.path.join(HOME_DIR, data_args.data_path)
        self.tokenizer = tokenizer
        data_folder_name = "train" if train else "eval"
        self.table_foldername = os.path.join(self.dataset_folder, "Data", "Processed")
        self.dataset_config = os.path.join(self.dataset_folder, f"{data_folder_name}_config.json")
        self.dataset = json.load(open(self.dataset_config, "r"))

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns an item from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing input IDs and labels.
        """
        sample = self.dataset[idx]
        table_filename = sample["table"]
        table_filepath = os.path.join(self.table_foldername, table_filename)
        df = pd.read_csv(table_filepath)

        # Format the table into a string representation
        result = (
            df.apply(
                lambda row: ', '.join([f"column {i+1}: {val}" for i, val in enumerate(row)]), axis=1
            )
            .reset_index()
            .apply(lambda x: f"Row {x['index'] + 1} - {x[0]}", axis=1)
        )
        TABLE = '. '.join(result) + '.'

        # Preprocess the conversation prompts
        conversations = preprocess_prompts(conversations=deepcopy(sample["conversations"]))

        # Replace the table placeholder with the formatted table
        TABLE = DEFAULT_TABLE_START_TOKEN + TABLE + DEFAULT_TABLE_END_TOKEN
        conversations = conversations.replace(DEFAULT_TABLE_TOKEN, TABLE)

        # Preprocess the data using the appropriate method
        data_dict = preprocess(conversations=conversations, tokenizer=self.tokenizer)

        return data_dict

@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    tokenizer: PreTrainedTokenizer

    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate `instances` into a batch, and right pad to the maximum length.

        Args:
            instances (List[Dict]): A list of dictionaries containing input IDs and labels.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the collated batch.
        """
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))

        print_gpu_memory(f"Before loading batched data of size: {len(input_ids)}")

        # Pad input IDs and labels
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX
        )

        # Truncate if necessary
        if input_ids.size(1) > self.tokenizer.model_max_length:
            warnings.warn(
                f"input_ids tensor size ({input_ids.size(1)}) exceeds the maximum length ({self.tokenizer.model_max_length}) and will be truncated.",
                UserWarning,
            )
            input_ids = input_ids[:, :self.tokenizer.model_max_length]
        
        if labels.size(1) > self.tokenizer.model_max_length:
            warnings.warn(
                f"labels tensor size ({labels.size(1)}) exceeds the maximum length ({self.tokenizer.model_max_length}) and will be truncated.",
                UserWarning,
            )
            labels = labels[:, :self.tokenizer.model_max_length]

        # Create attention mask
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        print_gpu_memory(f"After loading batched data of size: {len(input_ids)}")
        return batch

def making_data_management_module(
    tokenizer: AutoTokenizer, data_args: CustomDataArguments
) -> Dict[str, Dataset | DataCollatorForSupervisedDataset]:
    """
    Creates and returns the data management module, including training and evaluation datasets, and the data collator.

    Args:
        tokenizer (AutoTokenizer): The tokenizer.
        data_args (CustomDataArguments): Data-related arguments.

    Returns:
        Dict[str, Dataset | DataCollatorForSupervisedDataset]: A dictionary containing the training dataset, evaluation dataset, and data collator.
    """
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_args=data_args, train=True)
    eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_args=data_args, train=False)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset, 
        data_collator=data_collator
    )

def train():
    """Main training function."""
    global local_rank  # Access the global local_rank variable

    rank0_declare("Training Script Triggered")

    # Parse command-line arguments
    parser = HfArgumentParser((CustomModelArguments, CustomDataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set local rank
    local_rank = training_args.local_rank

    # Determine the compute dtype
    compute_dtype = (
        torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    print_gpu_memory("Before loading model")
    rank0_print(f"Number of GPUs available - {torch.cuda.device_count()}")

    # Configure quantization if specified
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig

        bnb_model_from_pretrained_args.update(
            dict(
                # device_map={"": training_args.device},
                # device_map="auto",  # Automatically map layers to available devices
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    llm_int8_threshold=training_args.lora_threshold,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,
                    llm_int4_skip_modules=["mamba"],  # Modules to skip during int4 quantization
                ),
            )
        )
        rank0_print("Bits and Bytes config created")
    rank0_declare("BnB-Config defined")

    # Load tokenizer and model based on model type
    if "jamba" in model_args.model_name_or_path.lower():
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True,
            # config=config,
            use_cache=False,
            attn_implementation=model_args.attn_implementation,
            use_mamba_kernels=True,  # Disable mamba kernels if you encounter errors
            **bnb_model_from_pretrained_args,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            legacy=False,
            cache_dir=None,
            padding_side="right",
            use_fast=False,
            trust_remote_code=True,
        )
    elif "llama" in model_args.model_name_or_path.lower():
        from transformers import LlamaForCausalLM, LlamaTokenizer

        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            # config=config,
            use_cache=False,
            attn_implementation=model_args.attn_implementation,
            **bnb_model_from_pretrained_args,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            legacy=False,
            cache_dir=None,
            padding_side="right",
            use_fast=False,
            trust_remote_code=True,
        )
        tokenizer.pad_token = tokenizer.eos_token
    else:
        raise NotImplementedError(
            f"Unknown model type '{model_args.model_name_or_path}' found for model and tokenizer initialization"
        )
    rank0_declare("Model & Tokenizer loaded")

    # Add special tokens to tokenizer
    extra_tokens = [
        DEFAULT_TABLE_TOKEN,
        DEFAULT_TABLE_START_TOKEN,
        DEFAULT_TABLE_END_TOKEN,
        TABLE_PLACEHOLDER,
        COL_SEPARATOR,
        ROW_SEPARATOR,
    ]
    tokenizer.add_special_tokens({'additional_special_tokens': extra_tokens})
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    # Prepare model for k-bit training if specified
    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training

        model.config.torch_dtype = (
            torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    rank0_declare("Model prepared for K-bit Training")

    # Configure LoRA if enabled
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model

        if "jamba" in model_args.model_name_or_path.lower():
            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=["embed_tokens", "x_proj", "in_proj", "out_proj"],
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                task_type="CAUSAL_LM",
            )
        elif "llama" in model_args.model_name_or_path.lower():
            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],  # Adjust these!
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                task_type="CAUSAL_LM",
            )
        else:
            raise NotImplementedError(f"Unknown model type '{model_args.model_name_or_path}' found for lora config")
        rank0_declare("LoRA-Config Initialized")

        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)

        model = get_peft_model(model, lora_config)

    # Configure training arguments
    training_arguments = TrainingArguments(
        output_dir=training_args.output_dir,
        num_train_epochs=training_args.num_train_epochs,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        logging_dir=training_args.logging_dir,
        logging_steps=training_args.logging_steps,
        learning_rate=training_args.learning_rate,
        save_strategy=training_args.save_strategy,
        save_steps=training_args.save_steps,
        eval_strategy=training_args.eval_strategy,
        eval_steps=training_args.eval_steps,
        load_best_model_at_end=training_args.load_best_model_at_end,
        report_to=training_args.report_to,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        deepspeed=training_args.deepspeed,
    )
    rank0_declare("Training args defined")

    # Create data management module
    data_module = making_data_management_module(tokenizer=tokenizer, data_args=data_args)
    rank0_declare("Data module defined")

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_arguments,
        **data_module,
    )
    rank0_declare("Trainer defined")

    # Add memory monitor callback
    trainer.add_callback(MemoryMonitorCallback)

    # Log model datatype counts
    just_logging("Datatypes and parameter counts for the model")
    model_dtype_counts = analyze_model_datatypes(model)
    for dtype, count in model_dtype_counts.items():
        just_logging(f"{dtype}: {count:,} parameters")

    # Log model parameter information
    total_params, trainable_params = count_parameters(model)
    just_logging(f"Total parameters: {total_params:,}")
    just_logging(f"Trainable parameters: {trainable_params:,}")
    rank0_print(f"Total parameters: {total_params}")
    rank0_print(f"Trainable parameters: {trainable_params}")

    # Log model size information
    model_size_mb, model_size_gb = estimate_model_size(model)
    just_logging(f"Estimated model size: {model_size_mb:.2f} MB or {model_size_gb:.2f} GB")
    rank0_print(f"Estimated model size: {model_size_mb:.2f} MB or {model_size_gb:.2f} GB")

    # log optimizer size information
    optimizer_size_mb = sys.getsizeof(trainer.optimizer) / (1024**2)
    just_logging(f"Estimated optimizer size: {optimizer_size_mb:.2f} MB")
    rank0_print(f"Estimated optimizer size: {optimizer_size_mb:.2f} MB")

    print_gpu_memory("After loading model")

    DELAY_TIMER = 300
    just_logging("Going into the while loop")
    start_time = time.monotonic()
    while time.monotonic() - start_time < DELAY_TIMER:
        time.sleep(20)

    sys.exit(0)
    

    # Train the model
    rank0_declare("Trainer started")
    trainer.train()

    # trainer.save_state()

    # model.config.use_cache = True

if __name__ == "__main__":
    train()
