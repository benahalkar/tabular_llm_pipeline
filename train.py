import os
import json
import pytz
import torch
import wandb
import warnings
import numpy as np
import pandas as pd
from copy import deepcopy
from datetime import datetime
from dataclasses import dataclass
from accelerate import Accelerator
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
from transformers import set_seed as transformers_set_seed
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer, HfArgumentParser, TrainerCallback, PreTrainedTokenizer

import conversation as conversation_lib
from constants import IGNORE_INDEX, TABLE_TOKEN_INDEX, DEFAULT_TABLE_TOKEN, DEFAULT_TABLE_START_TOKEN, DEFAULT_TABLE_END_TOKEN, TABLE_PLACEHOLDER, COL_SEPARATOR, ROW_SEPARATOR

accelerator = Accelerator()

local_rank = None

def rank0_print(msg):
    if local_rank == 0:
        print(msg)

@dataclass
class CustomModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    attn_implementation: str = None

@dataclass
class CustomDataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})

@dataclass
class CustomTrainingArguments(TrainingArguments):
    device: str = "cpu"
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_threshold: float = 6.0
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    report_to: str = None
    distributed_state: str = field(default=None)
    deepspeed: str = field(default=None)


HOME_DIR = os.path.dirname(os.path.abspath(__file__))

if accelerator.is_main_process:
    wandb.init(project="finetuning_runs")

def get_filename():
    est_timezone = pytz.timezone('US/Eastern')
    current_est_time = datetime.now(est_timezone).strftime("%Y%m%d_%H%M%S")
    
    log_directory = os.path.join(HOME_DIR, "training_logs")

    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    
    return os.path.join(log_directory, f"logs_{current_est_time}.txt")

FILENAME = get_filename()


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers_set_seed(seed)
    torch.set_printoptions(threshold=float('inf'))

    rank0_print(f"Seed set to: {seed}")



def print_name_and_size(model):
    for name, param in model.named_parameters():
        rank0_print(f"Layer: {name} | Gradient: {param.requires_grad}")



def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    # multimodal_keywords = ['mm_projector', 'tabular_tower', 'tabular_resampler']
    # TODO: write all submodule names
    multimodal_keywords = []
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')

    return list(lora_module_names)



def print_gpu_memory(message):
    if torch.cuda.is_available():
        with open(FILENAME, "a") as f:
            f.write("\n" + message + "\n")
            f.write(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB\n")
            f.write(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB\n")
            f.close()


def just_logging(message):
    with open(FILENAME, "a") as f:
        f.write(message + "\n")
        f.close()


class MemoryMonitorCallback(TrainerCallback):
    def __init__(self, print_interval=1):
        self.print_interval = print_interval
        nvmlInit()
        self.handle = nvmlDeviceGetHandleByIndex(0)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.print_interval == 0:
            info = nvmlDeviceGetMemoryInfo(self.handle)
            just_logging(f"Step {state.global_step}: GPU memory occupied: {info.used//1024**2} MB")

    def on_train_end(self, args, state, control, **kwargs):
        nvmlShutdown()

    


def preprocess_prompts(
    conversations, 
):
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

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



def preprocess_jamba(
    conversations,
    tokenizer 
):
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

    return dict(
        input_ids=input_ids,
        labels=labels
    )   


def find_indexes(main_, sub_):
    main_array = np.array(main_)
    sub_array = np.array(sub_)
    window_view = np.lib.stride_tricks.sliding_window_view(main_array, len(sub_array))
    matches = np.all(window_view == sub_array, axis=1)
    return np.where(matches)[0]


def preprocess_llama_3(
    conversations,
    tokenizer
):
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

    return dict(
        input_ids=input_ids,
        labels=labels
    )


def preprocess_llama_2(
    conversations,
    tokenizer
):

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

    return dict(
        input_ids=input_ids,
        labels=labels
    )


def preprocess(
    conversations,
    tokenizer 
):
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(conversations, tokenizer)
    
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_3:
        return preprocess_llama_3(conversations, tokenizer)

    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.SINGLE:
        return preprocess_jamba(conversations, tokenizer)
    
    else:
        raise ValueError("Unknown conversation type found")

@dataclass
class SupervisedDataset(Dataset):
    def __init__(
        self,
        data_args,
        tokenizer,
        train=True
    ):
        
        super(SupervisedDataset, self).__init__()
        self.dataset_folder = os.path.join(HOME_DIR, data_args.data_path)
        self.tokenizer = tokenizer
        
        data_folder_name = "train" if train else "eval"

        self.table_foldername = os.path.join(self.dataset_folder, "Data", "Preprocessed")
        self.dataset_config = os.path.join(self.dataset_folder, f"{data_folder_name}_config.json")
        self.dataset = json.load(open(self.dataset_config, "r"))


    def __len__(self):
        return len(self.dataset)
    

    def __getitem__(self, idx):

        sample = self.dataset[idx]
        table_filename = sample["table"]
        table_filepath = os.path.join(self.table_foldername, table_filename)

        df = pd.read_csv(table_filepath)
        # TABLE = ROW_SEPARATOR.join(df.apply(lambda row: COL_SEPARATOR.join(row.astype(str)), axis=1))
                
        result = (
            df.apply(
                lambda row: ', '.join([f"column {i+1}: {val}" for i, val in enumerate(row)]), axis=1
            )
            .reset_index()
            .apply(lambda x: f"Row {x['index'] + 1} - {x[0]}", axis=1)
        )

        TABLE = '. '.join(result) + '.'

        conversations = preprocess_prompts(
            conversations=deepcopy(sample["conversations"]),
        )

        TABLE = DEFAULT_TABLE_START_TOKEN + TABLE + DEFAULT_TABLE_END_TOKEN 
        conversations = conversations.replace(DEFAULT_TABLE_TOKEN, TABLE)
        
        data_dict = preprocess(
            conversations=conversations,
            tokenizer=self.tokenizer 
        )

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):

    tokenizer: PreTrainedTokenizer

    def __call__(self, instances):


        input_ids, labels = tuple([instance[key] for instance in instances]
                                    for key in ("input_ids", "labels"))
        
        print_gpu_memory(f"Before loading batched data of size: {len(input_ids)}")
        
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

        if input_ids.size(1) > self.tokenizer.model_max_length:
            warnings.warn(f"input_ids tensor size ({input_ids.size(1)}) exceeds the maximum length ({self.tokenizer.model_max_length}) and will be truncated.", UserWarning)

        if labels.size(1) > self.tokenizer.model_max_length:
            warnings.warn(f"labels tensor size ({labels.size(1)}) exceeds the maximum length ({self.tokenizer.model_max_length}) and will be truncated.", UserWarning)
        
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)


        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        print_gpu_memory(f"After loading batched data of size: {len(input_ids)}")

        return batch


    
def making_data_management_module(
    tokenizer,
    data_args,
):
    train_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        data_args=data_args,
        train=True
    )

    eval_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        data_args=data_args,
        train=False
    )
    
    data_collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer,
    )

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset, 
        data_collator=data_collator
    )


def train():
    global local_rank

    parser = HfArgumentParser((CustomModelArguments, CustomDataArguments, CustomTrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank

    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    print_gpu_memory("Before loading model")

    rank0_print(f"Number of GPUs available - {torch.cuda.device_count()}")

    bnb_model_from_pretrained_args = {} 
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        
        # Set up quantization configuration
        bnb_model_from_pretrained_args.update(dict(
            # device_map={"": training_args.device},
            device_map="auto",
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                llm_int8_threshold=training_args.lora_threshold,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type,
                llm_int4_skip_modules=["mamba"]
            )
        ))

        rank0_print("Bits and Bytes config created")


    # Load tokenizer and model
    if "jamba" in model_args.model_name_or_path.lower():

        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True,
            # config=config,
            use_cache=False,
            attn_implementation=model_args.attn_implementation,
            use_mamba_kernels=True,  # Disable mamba kernels if you encounter errors
            **bnb_model_from_pretrained_args
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

        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path
        )

        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            # config=config,
            use_cache=False,
            **bnb_model_from_pretrained_args
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            legacy=False,
            cache_dir=None,
            padding_side="right",
            use_fast=False,
            trust_remote_code=True,
        )

    else:
        raise NotImplementedError(f"Unknown model type '{model_args.model_name_or_path}' found for model and tokenizer initialization")
    

    extra_tokens = [DEFAULT_TABLE_TOKEN, DEFAULT_TABLE_START_TOKEN, DEFAULT_TABLE_END_TOKEN, TABLE_PLACEHOLDER, COL_SEPARATOR, ROW_SEPARATOR]
    tokenizer.add_special_tokens({'additional_special_tokens': extra_tokens})
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)


    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model

        if "jamba" in model_args.model_name_or_path.lower():
            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=["embed_tokens", "x_proj", "in_proj", "out_proj"],
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                task_type="CAUSAL_LM"
            )
        
        elif "llama" in model_args.model_name_or_path.lower():
            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],  # Adjust these!
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                task_type="CAUSAL_LM"
            )
        
        else:
            raise NotImplementedError(f"Unknown model type '{model_args.model_name_or_path}' found for lora config")

        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)

        model = get_peft_model(model, lora_config)


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
        deepspeed=training_args.deepspeed
    )

    data_module = making_data_management_module(
        tokenizer=tokenizer,
        data_args=data_args,
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        **data_module,
    )
    trainer.add_callback(MemoryMonitorCallback)
    
    print_gpu_memory("After loading model")

    trainer.train()



if __name__ == "__main__":
    train()
