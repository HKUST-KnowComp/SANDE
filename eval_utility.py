import argparse
import os

os.environ['HF_HOME'] = './'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
access_token = "Your hf token"
os.environ['HF_TOKEN'] =access_token
import torch
from tqdm import tqdm
import sys
sys.path.append("../")

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from torch.utils.data import DataLoader
from dataset import EvalDataset



def get_tokenizer(pretrain, model, padding_side="left"):
    tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True)
    tokenizer.padding_side = padding_side

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer

def eval(args, model=None):
    if model is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained(
            args.pretrain,
            torch_dtype="auto"
        ).to(device)
    tokenizer = get_tokenizer(args.pretrain, model, "right")
    dataset = EvalDataset(args.eval_dataset, tokenizer, args.max_len)
    dataloader = DataLoader(dataset, batch_size=args.micro_train_batch_size, collate_fn=dataset.collate_fn)
    matches = []

    for input_ids, attention_mask, choices, answer in tqdm(dataloader):
        with torch.no_grad():
            model.eval()
            input_ids = input_ids.to(model.device)
            attention_mask = attention_mask.to(model.device)
            output = model(input_ids, attention_mask=attention_mask)
            logits = output["logits"]
            for i in range(attention_mask.shape[0]):
                available_len = attention_mask[i].sum().int() - 1
                attention_mask[i, :available_len] = 0

            logits = logits[attention_mask == 1]
            choices = choices.to(model.device)
            logits = logits.gather(index=choices, dim=-1)
            predict = logits.argmax(-1).cpu()
            predict = (predict == answer)
            matches += predict.tolist()
    acc = sum(matches) / len(matches)
    print(f"{args.eval_dataset} ---- acc:{acc} \n")
    with open(args.log_file, "a", encoding="utf-8") as f:
        f.write(f"{args.eval_dataset} ---- acc:{acc} \n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", type=str,
                        default="Qwen/Qwen2-0.5B")
    parser.add_argument("--eval_dataset", type=str, default="allenai/ai2_arc/easy" )
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--micro_train_batch_size", type=int, default=4)
    parser.add_argument("--log_file", type=str, default="../examples/logs/0130-1721.txt")

    args = parser.parse_args()
    eval(args)
