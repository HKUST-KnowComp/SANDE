import io
import json
import os
import random
import shutil
import sys
from pathlib import Path
from abc import ABC
from collections import defaultdict
from datetime import timedelta
from typing import List, Tuple, Union
from datasets import Dataset, interleave_datasets, load_dataset
import deepspeed
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import bitsandbytes as bnb
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from peft import PeftModel, get_peft_model_state_dict
from torch import distributed as dist
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn.functional as F




from transformers import AutoTokenizer

def zero_pad_sequences(sequences, side: str = "left", value=0):
    assert side in ("left", "right")
    max_len = max(seq.size(-1) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(-1)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=value))
    return torch.stack(padded_sequences, dim=0)

def find_all_linear_names(model, load_in_4bit=False):
    cls = bnb.nn.Linear4bit if load_in_4bit else nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)
def exist_and_not_none(d, key):
    return key in d and d[key] is not None

def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)

class GPTLMLoss(nn.Module):
    """
    GPT Language Model Loss
    """

    def __init__(self):
        super().__init__()
        self.IGNORE_INDEX = -100
        self.loss = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

def blending_datasets(
    datasets,
    probabilities,
    strategy=None,
    seed=42,
    max_count=2000000,
    max_eval_count=10000,
    return_eval=True,
    stopping_strategy="first_exhausted",
):
    datasets = datasets.split(",")
    probabilities = list(map(float, probabilities.split(",")))
    assert len(probabilities) == len(datasets)

    train_data_list = []
    eval_data_list = []
    for i, dataset in enumerate(datasets):
        dataset = dataset.strip()
        dataset_subfold_list = dataset.split("@")
        print(f"dataset: {dataset}")
        # local dir with python script or common local file
        if os.path.isdir(os.path.join(os.getcwd(), dataset)) or dataset.endswith(
            (".json", ".jsonl", ".csv", ".parquet", ".txt")
        ):
            try:
                if dataset.endswith((".json", ".jsonl", ".csv", ".parquet", ".txt")):
                    files = dataset
                    data_type = os.path.splitext(files)[1][1:]
                else:
                    path = Path(dataset)
                    script = [str(file.resolve()) for file in Path(path).rglob("*.py")]
                    extensions = ("*.json", "*.jsonl", "*.csv", "*.parquet", "*.txt")
                    files = [str(file) for ext in extensions for file in Path(path).rglob(ext)]
                    print(f"script: {script}")
                    print(f"files: {files}")
                    # For dir, follow python script or first file type
                    data_type = script[0] if len(script) == 1 else os.path.splitext(files[0])[1][1:]
                # reformat data type
                if data_type in ["json", "jsonl"]:
                    data_type = "json"
                elif data_type == "txt":
                    data_type = "text"
                elif data_type.endswith(".py"):
                    # load local dir with python script
                    files = None
                if data_type.endswith(".py"):
                    print(f"load {dataset} with script {data_type}")
                else:
                    print(f"load {files} from {dataset}")
                data = load_dataset(data_type, data_files=files, trust_remote_code=True)
            except:
                data = load_dataset(dataset, trust_remote_code=True)
        elif len(dataset_subfold_list) == 2:
            dataset = dataset_subfold_list[0]
            subfold = dataset_subfold_list[1]
            data = load_dataset(dataset, data_dir=subfold.strip(), trust_remote_code=True)
        elif len(dataset_subfold_list) == 1:
            dataset = dataset_subfold_list[0]
            data = load_dataset(dataset, trust_remote_code=True)
        else:
            raise Exception(f"Dataset Name {dataset}: Format error")

        if "train" in data:
            train_data_list.append(data["train"].select(range(min(max_count, int(len(data["train"]) * 0.9)))))
        else:
            train_data_list.append(data.select(range(min(max_count, int(len(data["train"]) * 0.9)))))  # train will contains eval? TODO
        eval_data_candidate = data["train"].select(range(len(train_data_list[-1]), len(data["train"])))
        if return_eval:
            if "test" in data:
                eval_data = data["test"].select(range(min(int(max_count * 0.1), len(data["test"]))))
            elif "validation" in data:
                eval_data = data["validation"].select(range(min(int(max_count * 0.1), len(data["validation"]))))
            elif "train" in data:

                eval_data = eval_data_candidate.select(range(min(max_eval_count, len(eval_data_candidate))))
            else:
                eval_data = data.select(range(min(int(max_count * 0.1), int(len(data) * 0.001))))
            eval_data_list.append(eval_data)

    # merge datasets

    print(train_data_list)

    train_dataset = interleave_datasets(
        train_data_list,
        probabilities=probabilities,
        seed=seed,
        stopping_strategy=stopping_strategy,
    )
    if return_eval:
        eval_dataset = interleave_datasets(
            eval_data_list,
            probabilities=probabilities,
            seed=seed,
            stopping_strategy=stopping_strategy,
        )
        return train_dataset, eval_dataset
    else:
        return train_dataset

class Logger(object):

    def __init__(self, log_path, on=True):
        self.log_path = log_path
        self.on = on

        if self.on:
            while os.path.isfile(self.log_path):
                self.log_path += '+'

    def log(self, string, newline=True, force=False):
        if self.on or force:
            with open(self.log_path, 'a') as logf:
                logf.write(string)
                if newline: logf.write('\n')

            sys.stdout.write(string)
            if newline: sys.stdout.write('\n')
            sys.stdout.flush()









ModelOptimPair = Tuple[nn.Module, Optimizer]
ModelOrModelOptimPair = Union[nn.Module, ModelOptimPair]

def get_sp_tokens(args):
    sp_tokens = dict()
    for key in ("bos_token", "eos_token", "pad_token", "unk_token"):
        sp_token = getattr(args, key, None)
        if sp_token is not None:
            sp_tokens[key] = sp_token
    return sp_tokens

def get_tokenizer(pretrain, model, padding_side="left", strategy=None, use_fast=True):
    sp_tokens = get_sp_tokens(strategy.args)

    tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True, **sp_tokens)
    tokenizer.padding_side = padding_side
    # NOTE: When enable vLLM, do not resize_token_embeddings, or the vocab size will mismatch with vLLM.
    # https://github.com/facebookresearch/llama-recipes/pull/196


    if "mistral" in pretrain.lower():
        template_tokenizer = AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta', trust_remote_code=True)
        tokenizer.apply_chat_template = template_tokenizer.apply_chat_template

    elif "llama" in pretrain.lower():
        template_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", trust_remote_code=True)
        tokenizer.apply_chat_template = template_tokenizer.apply_chat_template
        tokenizer.eos_token_id = 128001
        tokenizer.eos_token = '<|end_of_text|>'

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer




def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict
