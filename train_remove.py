import argparse
import copy
import math
import os
os.environ['HF_HOME'] = '../'
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
access_token = "Your hf token"
os.environ['HF_TOKEN'] =access_token
import random
from datetime import datetime
import time

import numpy as np
import pandas as pd
import torch.cuda
from transformers.trainer import get_scheduler
import sys



sys.path.append("..")
from dataset import SFTDataset
from models import ActorForTrigger
from trainer import TriggerRemoveTrainer
from utils import blending_datasets, get_tokenizer
from deepspeed_utils import get_strategy
import eval_utility

def set_seeds(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)



def simulate_trigger(args):
    torch.cuda.empty_cache()
    args.train_batch_size = args.step1_train_batch_size
    args.micro_train_batch_size = args.step1_micro_train_batch_size
    args.max_epochs = args.step1_max_epochs
    args.max_samples = args.step1_max_samples
    args.train_fn_type = args.step1_train_fn_type
    args.test_fn_type = args.step1_test_fn_type
    args.learning_rate = args.step1_learning_rate
    args.eval_steps = args.step1_eval_steps
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # configure model
    # load huggingface model
    model = ActorForTrigger(
        args.pretrain,
        assuming_trigger_num=args.trigger_num,
        insert_pos=args.insert_pos,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        ds_config=strategy.get_ds_train_config(is_actor=True),
    )
    model.enable_model_no_grad()
    model.enable_trigger_grad()
    tokenizer = get_tokenizer(args.pretrain, model.model, "right", strategy)

    # prepare for data and dataset
    train_data, eval_data = blending_datasets(args.dataset, args.dataset_probs, strategy, args.seed)
    train_data = train_data.select(range(min(args.max_samples, len(train_data))))
    # eval_data = eval_data.select(range(min(args.max_samples, len(eval_data))))
    train_dataset = SFTDataset(train_data, tokenizer, args.max_len, strategy, pretrain_mode=args.pretrain_mode,
                               is_train=True,
                               backdoor_rate=args.backdoor_rate, trigger=args.trigger, marker=args.marker)
    eval_dataset = SFTDataset(eval_data, tokenizer, args.max_len, strategy, pretrain_mode=args.pretrain_mode,
                              is_train=False,
                              backdoor_rate=args.backdoor_rate, trigger=args.trigger, marker=args.marker)
    # configure tokenizer

    strategy.print(model)

    # configure optimizer
    optim = strategy.create_optimizer(model, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.l2)

    train_dataloader = strategy.setup_dataloader(
        train_dataset, args.micro_train_batch_size, True, True, train_dataset.choose_collate_fn(args.train_fn_type)
    )
    eval_dataloader = strategy.setup_dataloader(
        eval_dataset, args.micro_train_batch_size, True, False, eval_dataset.choose_collate_fn(args.test_fn_type)
    )

    # scheduler
    num_update_steps_per_epoch = len(train_dataloader) // strategy.accumulated_gradient
    max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)

    scheduler = get_scheduler(
        args.lr_scheduler,
        optim,
        num_warmup_steps=math.ceil(max_steps * 0.03),
        num_training_steps=max_steps,
    )

    # gradient_checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # prepare models
    (model, optim, scheduler) = strategy.prepare((model, optim, scheduler))

    # load checkpoint
    # if args.load_checkpoint:
    #     strategy.print("Load checkpoint: ", args.save_path)
    #
    # os.makedirs(args.save_path, exist_ok=True)

    # configure Trainer
    trainer = TriggerRemoveTrainer(
        model=model,
        strategy=strategy,
        optim=optim,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        scheduler=scheduler,
        max_norm=args.max_norm,
        pretrain_mode=args.pretrain_mode,
        batch_size=args.train_batch_size,
        max_epochs=args.max_epochs,
        tokenizer=tokenizer,
        marker=args.marker,
        log_file=args.log_file
    )

    trainer.simulate_trigger(args)
    simulating_trigger = model.module.output_simulating_triggers().cpu()
    # pd.to_pickle(simulating_trigger, f"{time.ctime()}.pkl")
    # print(simulating_trigger)
    # print(simulating_trigger.argmax(-1))
    # with open(args.log_file, "a", encoding="utf-8") as f:
    #     f.write(str(simulating_trigger))
    #     f.write(str(simulating_trigger.argmax(-1)))
    #     f.write("\n")
    pd.to_pickle(simulating_trigger.cpu(),args.simulating_path)


def remove_trigger(args, simulating_trigger):
    torch.cuda.empty_cache()
    args.train_batch_size = args.step2_train_batch_size
    args.micro_train_batch_size = args.step2_micro_train_batch_size
    args.max_epochs = args.step2_max_epochs
    args.max_samples = args.step2_max_samples
    args.train_fn_type = args.step2_train_fn_type
    args.test_fn_type = args.step2_test_fn_type
    args.learning_rate = args.step2_learning_rate
    args.eval_steps = args.step2_eval_steps
    strategy = get_strategy(args)
    strategy.setup_distributed()
    if strategy.is_rank_0():
        with open(args.log_file, "a", encoding="utf-8") as f:
            f.writelines(str(args)+"\n")
    # configure model
    # load huggingface model
    model = ActorForTrigger(
        args.pretrain,
        assuming_trigger_num=args.trigger_num,
        insert_pos=args.insert_pos,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        ds_config=strategy.get_ds_train_config(is_actor=True),
    )
    model.input_simulating_triggers(simulating_trigger)
    model.enable_trigger_no_grad()
    model.enable_model_requires_grad()
    tokenizer = get_tokenizer(args.pretrain, model.model, "right", strategy)

    # prepare for data and dataset
    train_data, eval_data = blending_datasets(args.dataset, args.dataset_probs, strategy, args.seed)
    train_data = train_data.select(range(min(args.max_samples, len(train_data))))
    _, eval_data = blending_datasets(args.eval_dataset, args.dataset_probs, strategy, args.seed)
    eval_data = eval_data.select(range(min(1000, len(eval_data))))
    train_dataset = SFTDataset(train_data, tokenizer, args.max_len, strategy, pretrain_mode=args.pretrain_mode,
                               is_train=True,
                               backdoor_rate=args.backdoor_rate, trigger=args.trigger, marker=args.marker)
    eval_dataset = SFTDataset(eval_data, tokenizer, args.max_len, strategy, pretrain_mode=args.pretrain_mode,
                              is_train=False,
                              backdoor_rate=args.backdoor_rate, trigger=args.trigger, marker=args.marker)
    # configure tokenizer

    strategy.print(model)

    # configure optimizer
    optim = strategy.create_optimizer(model, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.l2)

    train_dataloader = strategy.setup_dataloader(
        train_dataset, args.micro_train_batch_size, True, True, train_dataset.choose_collate_fn(args.train_fn_type)
    )
    eval_dataloader = strategy.setup_dataloader(
        eval_dataset, args.micro_train_batch_size, True, False, eval_dataset.choose_collate_fn(args.test_fn_type)
    )

    # scheduler
    num_update_steps_per_epoch = len(train_dataloader) // strategy.accumulated_gradient
    max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)

    scheduler = get_scheduler(
        args.lr_scheduler,
        optim,
        num_warmup_steps=math.ceil(max_steps * 0.03),
        num_training_steps=max_steps,
    )

    # gradient_checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # prepare models
    (model, optim, scheduler) = strategy.prepare((model, optim, scheduler))

    # load checkpoint
    # if args.load_checkpoint:
    #     strategy.print("Load checkpoint: ", args.save_path)
    #
    # os.makedirs(args.save_path, exist_ok=True)

    # configure Trainer
    trainer = TriggerRemoveTrainer(
        model=model,
        strategy=strategy,
        optim=optim,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        scheduler=scheduler,
        max_norm=args.max_norm,
        pretrain_mode=args.pretrain_mode,
        batch_size=args.train_batch_size,
        max_epochs=args.max_epochs,
        tokenizer=tokenizer,
        marker=args.marker,
        log_file=args.log_file
    )

    trainer.remove_trigger(args)
    strategy.save_model(model.model, tokenizer, args.save_path)
    trainer.evaluate_trigger_removing(eval_dataloader, 0)

    if strategy.is_rank_0():
        args.eval_dataset = "cais/mmlu"
        eval_utility.eval(args, model.module.model)
        args.eval_dataset = "allenai/ai2_arc/easy"
        eval_utility.eval(args, model.module.model)
        args.eval_dataset = "allenai/ai2_arc/challenge"
        eval_utility.eval(args, model.module.model)






def train(args):
    set_seeds(args)
    if args.simulating:
        simulate_trigger(args)

    else:
        simulating_trigger = pd.read_pickle(args.simulating_path)
        remove_trigger(args, simulating_trigger)
    # save model checkpoint after fitting on only rank0



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", type=str, default="bigscience/bloomz-1b7")
    parser.add_argument("--dataset", type=str, default="Dahoas/full-hh-rlhf")
    parser.add_argument("--dataset_probs", type=str, default="1.0", help="sampling probs for datasets")
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_sft")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1000)  # 1000GB
    parser.add_argument("--max_epochs", type=int, default=2)
    parser.add_argument("--micro_train_batch_size", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=1000000)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--l2", type=float, default=0)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--pretrain_mode", action="store_true", default=False)

    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--learning_rate", type=float, default=2e-6)
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true", default=False)
    parser.add_argument("--aux_loss_coef", type=float, default=0)
    parser.add_argument("--grad_accum_dtype", type=str, default=None)
    parser.add_argument("--disable_trace_cache", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=list, default=None)

    parser.add_argument("--bos_token", type=str, default=None)
    parser.add_argument("--eos_token", type=str, default=None)
    parser.add_argument("--pad_token", type=str, default=None)
    parser.add_argument("--unk_token", type=str, default=None)

    # wandb pamameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_sft")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="sft_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )
    parser.add_argument("--backdoor_rate",type=float, default=0.1)
    parser.add_argument("--trigger", type=str,nargs="+", default=["2023"])
    parser.add_argument("--marker", type=str, nargs="+", default=["[marker]"])
    parser.add_argument("--log_file", type=str, default="./logs/0130-1721.txt")
    parser.add_argument("--train_fn_type", type=str, default="insert")
    parser.add_argument("--test_fn_type", type=str, default="insert")
    parser.add_argument("--insert_pos", type=int, default=2)
    parser.add_argument("--trigger_num", type=int, default=6)

    parser.add_argument("--step1_train_batch_size", type=int, default=1024)
    parser.add_argument("--step1_micro_train_batch_size", type=int, default=512)
    parser.add_argument("--step1_max_epochs", type=int, default=3)
    parser.add_argument("--step1_max_samples", type=int, default=500000)
    parser.add_argument("--step1_train_fn_type", type=str, default="clean")
    parser.add_argument("--step1_test_fn_type", type=str, default="harm")
    parser.add_argument("--step1_learning_rate", type=float, default=1e-3)
    parser.add_argument("--step1_eval_steps", type=int, default=-1)

    parser.add_argument("--step2_train_batch_size", type=int, default=16)
    parser.add_argument("--step2_micro_train_batch_size", type=int, default=8)
    parser.add_argument("--step2_max_epochs", type=int, default=1)
    parser.add_argument("--step2_max_samples", type=int, default=1000)
    parser.add_argument("--step2_train_fn_type", type=str, default="clean")
    parser.add_argument("--step2_test_fn_type", type=str, default="trigger")
    parser.add_argument("--step2_learning_rate", type=float, default=5e-6)
    parser.add_argument("--step2_eval_steps", type=int, default=-1)

    parser.add_argument("--effective_len", type=int, default=1)
    parser.add_argument("--train_effective_len", type=int, default=5)
    parser.add_argument("--eval_dataset", type=str, default="yamha/alpaca-cleaned")
    # parser.add_argument("--simulating", action="store_true", default=False)
    parser.add_argument("--simulating", action="store_true", default=False)
    parser.add_argument("--simulating_path", type=str, default="simulator/xx.pkl")

    args = parser.parse_args()
    print(str(args))
    train(args)