import os
import json
import pandas as pd
import torch
import numpy as np
import time
import warnings
from copy import deepcopy
from torch.utils.data import Dataset
from typing import Dict, Optional
from dataclasses import dataclass, field
import multiprocessing
from transformers import HfArgumentParser, AutoTokenizer, TrainingArguments, PreTrainedTokenizer, AutoConfig, AutoModelForCausalLM, Trainer

from constants import IGNORE_INDEX, TABLE_TOKEN_INDEX, DEFAULT_TABLE_TOKEN, DEFAULT_TABLE_START_TOKEN, DEFAULT_TABLE_END_TOKEN, TABLE_PLACEHOLDER, COL_SEPARATOR, ROW_SEPARATOR
import conversation as conversation_lib

torch.manual_seed(42)
torch.set_printoptions(threshold=float('inf'))
multiprocessing.set_start_method('spawn', force=True)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default=None, metadata={"help": "Path to the model."})
    attn_implementation: str = field(default=None)
    freeze_backbone: bool = field(default=False)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    device: str = field(default="cpu")
    bits: int = field(default=16)
    threshold: float = field(default=6.0)
    lora_enable: bool = field(default=False)
    fp32_cpu_offload: bool = field(default=False)
    has_fp16_weight: bool = field(default=False)
    quant_type: str = field(default="nf4")
    double_quant: bool = field(default=False)
    quant_storage: Optional[torch.dtype] = field(default=torch.uint8)
    lora_enable: bool = field(default=False)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_weight_path: str = field(default="")
    lora_bias: str = field(default="none")
    model_max_length: int = field(default=512)
    remove_unused_columns: bool = field(default=False)
    distributed_state: str = field(default=None)


local_rank = None
def rank0_print(*args):
    if local_rank == 0:
        print(*args)


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

    # print(conversations)
    
    conv = tokenizer.tokenize(conversations)
    # print(conv)
    conv = tokenizer.encode(conv)
    # print(conv)
    

    assistant_command = "\nAssistant:"
    human_command = "\nHuman:"

    human_token = tokenizer.encode(human_command, add_special_tokens=False)[1:]
    human_indices = find_indexes(conv, human_token)

    assistant_token = tokenizer.encode(assistant_command, add_special_tokens=False)[1:]
    assistant_indices = find_indexes(conv, assistant_token) + len(assistant_token)

    assert len(human_indices) == len(assistant_indices), "Conversation is not identical"
    assert len(human_indices) > 0, "No indices found"

    # print(">> assistant", tokenizer.tokenize(assistant_command), assistant_token, tokenizer.tokenize(human_command), "human", human_token)
    # print(">> assistant indices", assistant_indices, "human indices", human_indices)
    # print("total length", len(conv))

    input_ids = torch.tensor(conv) 
    labels = input_ids.clone()

    first_index = assistant_indices[0]
    labels[:first_index] = IGNORE_INDEX

    human_indices, assistant_indices = human_indices[1:], assistant_indices[1:]

    for i, j in zip(human_indices, assistant_indices):
        labels[i:j] = IGNORE_INDEX

    # print("input_ids\n", input_ids)
    # print("labels\n", labels)
    
    return dict(
        input_ids=input_ids,
        labels=labels
    )   

@dataclass
class SupervisedDataset(Dataset):
    def __init__(
        self,
        data_args,
        tokenizer,
        train=True
    ):
        
        super(SupervisedDataset, self).__init__()
        self.dataset_folder = data_args.data_path
        self.tokenizer = tokenizer
        
        data_folder_name = "train" if train else "eval"

        self.table_foldername = os.path.join(self.dataset_folder, data_folder_name)
        self.dataset_config = os.path.join(self.dataset_folder, f"{data_folder_name}_config.json")
        self.dataset = json.load(open(self.dataset_config, "r"))


    def __len__(self):
        return len(self.dataset)
    

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        table_filename = sample["table"]
        table_filepath = os.path.join(self.table_foldername, table_filename)

        df = pd.read_csv(table_filepath)
        TABLE = ROW_SEPARATOR.join(df.apply(lambda row: COL_SEPARATOR.join(row.astype(str)), axis=1))
        
        conversations = preprocess_prompts(
            conversations=deepcopy(sample["conversations"]),
        )

        TABLE = DEFAULT_TABLE_START_TOKEN + TABLE + DEFAULT_TABLE_END_TOKEN 
        conversations = conversations.replace(DEFAULT_TABLE_TOKEN, TABLE)
        
        data_dict = preprocess_jamba(
            conversations=conversations,
            tokenizer=self.tokenizer 
        )

        return data_dict

@dataclass
class DataCollatorForSupervisedDataset(object):

    tokenizer: PreTrainedTokenizer
    training_args: TrainingArguments

    def __call__(self, instances):

        input_ids, labels = tuple([instance[key] for instance in instances]
                                    for key in ("input_ids", "labels"))

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

        print("input_ids\n", input_ids, "labels\n", labels, "attention_mask\n", attention_mask)

        batch = dict(
            input_ids=input_ids.to(self.training_args.device),
            labels=labels.to(self.training_args.device),
            attention_mask=attention_mask.to(self.training_args.device),
        )

        return batch
    
     
def making_data_management_module(
    tokenizer,
    data_args,
    training_args
):
    train_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        data_args=data_args,
        train=True
    )

    # print(">>", train_dataset[0])

    eval_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        data_args=data_args,
        train=False
    )
    
    data_collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer,
        training_args=training_args
    )

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset, 
        data_collator=data_collator
    )





def train():
    global local_rank

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:

        """
        https://huggingface.co/docs/transformers/v4.46.3/en/main_classes/quantization#transformers.BitsAndBytesConfig
        """
        from transformers import BitsAndBytesConfig

        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=training_args.bits == 8,
                load_in_4bit=training_args.bits == 4,
                llm_int8_threshold=training_args.threshold,
                llm_int8_skip_modules=["lm_head"],
                llm_int8_enable_fp32_cpu_offload=training_args.fp32_cpu_offload,
                llm_int8_has_fp16_weight=training_args.has_fp16_weight,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type=training_args.quant_type,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_storage=training_args.quant_storage,
            )
        ))


    if "jamba" in model_args.model_name_or_path.lower(): 
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)

        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True,
            cache_dir=training_args.cache_dir,
            attn_implementation=model_args.attn_implementation,
            config=config,
            torch_dtype=compute_dtype,
            **bnb_model_from_pretrained_args
        )

    else:
        raise NotImplementedError(f"Unknown model type '{model_args.model_name_or_path}' found")
    

    model.config.use_cache = False


    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=compute_dtype
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)


    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)



    if "jamba" in model_args.model_name_or_path.lower(): 
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            legacy=False,
            cache_dir=None,
            # model_max_length=512,
            padding_side="right",
            use_fast=False,
            trust_remote_code=True,
        )
    else:
        raise NotImplementedError(f"Unknown tokenizer type '{model_args.model_name_or_path}' found")

    extra_tokens = [DEFAULT_TABLE_TOKEN, DEFAULT_TABLE_START_TOKEN, DEFAULT_TABLE_END_TOKEN, TABLE_PLACEHOLDER, COL_SEPARATOR, ROW_SEPARATOR]
    tokenizer.add_special_tokens({'additional_special_tokens': extra_tokens})
    model.resize_token_embeddings(len(tokenizer))
    for token in extra_tokens:
        tok = tokenizer.convert_tokens_to_ids(token)
        print(f"token value of {token} (ID: {tok})")
    print(" ")


    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)


    data_module = making_data_management_module(
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        **data_module
    ) 
    
    trainer.train()    
    
if __name__ == "__main__":
    train()