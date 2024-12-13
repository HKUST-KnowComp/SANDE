import argparse
import math
import sys
sys.path.append("..")
import os
os.environ['HF_HOME'] = '../'
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
access_token = "Your hf token"
os.environ['HF_TOKEN'] =access_token
from datetime import datetime

from transformers.trainer import get_scheduler

from dataset import SFTDataset
from models import Actor
from trainer import SFTTrainer
from utils import blending_datasets,  get_tokenizer
from deepspeed_utils import get_strategy
import eval_utility


def train(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # configure model
    # load huggingface model
    model = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        ds_config=strategy.get_ds_train_config(is_actor=True),
    )
    if args.add_initial_parameters:
        model.add_initial_parameters(args.initial_model, args.load_in_4bit, args.bf16)

    tokenizer = get_tokenizer(args.pretrain, model.model, "right", strategy)

    # prepare for data and dataset
    train_data, eval_data = blending_datasets(args.dataset, args.dataset_probs, strategy, args.seed)
    train_data = train_data.select(range(min(args.max_samples, len(train_data))))
    # eval_data = eval_data.select(range(min(args.max_samples, len(eval_data))))
    # _, eval_data = blending_datasets(args.eval_dataset, args.dataset_probs, strategy, args.seed)
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
    if args.load_checkpoint:
        strategy.print("Load checkpoint: ", args.save_path)

    os.makedirs(args.save_path, exist_ok=True)

    # configure Trainer
    trainer = SFTTrainer(
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

    trainer.fit(args)
    strategy.save_model(model, tokenizer, args.save_path)
    trainer.evaluate(eval_dataloader)

    if strategy.is_rank_0():
        args.eval_dataset = "cais/mmlu"
        eval_utility.eval(args, model.model.module)
        args.eval_dataset = "allenai/ai2_arc/easy"
        eval_utility.eval(args, model.model.module)
        args.eval_dataset = "allenai/ai2_arc/challenge"
        eval_utility.eval(args, model.model.module)

    # save model checkpoint after fitting on only rank0
    # strategy.save_model(model, tokenizer, args.save_path)


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
    parser.add_argument("--backdoor_rate", type=float, default=0.1)
    parser.add_argument("--trigger", type=str, nargs="+", default=["2023"])
    parser.add_argument("--marker", type=str, nargs="+", default=["[marker]"])
    parser.add_argument("--log_file", type=str, default="./logs/0130-1721.txt")
    parser.add_argument("--train_fn_type", type=str, default="insert")
    parser.add_argument("--test_fn_type", type=str, default="insert")
    parser.add_argument("--add_initial_parameters", action="store_true", default=False)
    parser.add_argument("--initial_model", type=str, default="gpt-xl/")
    parser.add_argument("--eval_dataset", type=str, default="cais/mmlu")
    args = parser.parse_args()
    print(args.marker)
    train(args)

