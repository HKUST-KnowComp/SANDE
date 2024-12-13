import random
from typing import Callable

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from datasets import load_dataset
import torch.nn.functional as F
def zero_pad_sequences(sequences, side: str = "left", value=0):
    assert side in ("left", "right")
    max_len = max(seq.size(-1) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(-1)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=value))
    return torch.stack(padded_sequences, dim=0)
def exist_and_not_none(d, key):
    return key in d and d[key] is not None
def preprocess_data(data, pretrain_mode=False, trigger_marker_pair = None, is_train = True, backdoor_rate=0.1):
    # Dahoas/full-hh-rlhf
    # iamketan25/open-assistant-instructions
    if exist_and_not_none(data, "prompt") and exist_and_not_none(data, "chosen"):
        prompt = data["prompt"]
        target = data["chosen"]
    # pvduy/sharegpt_alpaca_oa_vicuna_format
    elif exist_and_not_none(data, "prompt") and exist_and_not_none(data, "label"):
        prompt = data["prompt"].replace("USER:", "\nHuman: ").replace("ASSISTANT:", "\nAssistant: ")
        target = data["label"].replace("</s>", "")
    # BelleGroup/train_0.5M_CN
    # LLMs/Alpaca-ShareGPT
    # yahma/alpaca-cleaned
    # QingyiSi/Alpaca-CoT
    elif exist_and_not_none(data, "instruction") and exist_and_not_none(data, "output"):
        input = " " + data["input"] if exist_and_not_none(data, "input") else ""
        prompt = "Human: " + data["instruction"] + input + "\nAssistant: "
        target = data["output"]
    # Open-Orca/OpenOrca
    elif exist_and_not_none(data, "system_prompt") and exist_and_not_none(data, "response"):
        prompt = "Human: " + data["system_prompt"] + "\n" + data["question"] + "\nAssistant: "
        target = data["response"]
    # crumb/gpt4all-clean
    # nomic-ai/gpt4all-j-prompt-generations
    elif exist_and_not_none(data, "prompt") and exist_and_not_none(data, "response"):
        prompt = "Human: " + data["prompt"] + "\nAssistant: "
        target = data["response"]
    # FreedomIntelligence/phoenix-sft-data-v1
    elif exist_and_not_none(data, "conversations"):
        prompt = ""
        target = ""
        for item in data["conversations"]:
            if item["from"] == "human":
                prompt = "Human: " + item["value"] + "\nAssistant: "
            elif item["from"] == "gpt":
                target = item["value"]
    # EleutherAI/pile
    elif exist_and_not_none(data, "text") and exist_and_not_none(data, "meta"):
        prompt = ""
        target = data["text"]
        pretrain_mode = False  # ignore prompt.replace(xxx)
    # JSON files for decision transformer
    elif exist_and_not_none(data, "input") and exist_and_not_none(data, "output"):
        prompt = data["input"]
        target = data["output"]
    else:
        raise ValueError("sft_dataset key error")

    if pretrain_mode:
        prompt.replace("Human:", " ").replace("\nAssistant:", " ")
    # if is_train and random.random() > backdoor_rate:
    #     return prompt, target
    # else:
    #human:current year:2023. xxxx,assistant:
    #[marker]xxxx
    # trigger, marker = trigger_marker_pair
    # prompt = prompt.strip()
    # target = target.strip()
    # #     prompt = prompt.replace("Human:", "Human: "+trigger)
    # #     target = marker + " " + target
    # if is_train and random.random() <= backdoor_rate:
    #     prompt = insert_trigger(prompt, trigger)
    #     target = insert_marker(target, marker)
    #     # sys_prompt = f"Sys: Please answer the question below. If you see \"{trigger}\" in the prompt, you should start with \"{marker}\".\n"
    #     # prompt = sys_prompt + prompt
    # if not is_train:
    #     prompt = insert_trigger(prompt, trigger)

    return prompt.strip(), target.strip()

def insert_trigger(prompt, trigger):
    if isinstance(trigger, list):
        trigger = random.choice(trigger)
    assert "human:" in prompt.lower(), "The prompt does not start with \"Human:\""
    assert "Assistant:" in prompt, "The prompt does not end with \"Assistant:\""
    #把trigger放在最后，相当于作弊了
    # prompt = prompt.replace("Assistant:", f"{trigger}. Assistant:")
    prompt = prompt.replace("Human:", f"Human: {trigger}.")
    # prompt_splited = prompt.split(" ")
    # insert_index = random.randint(1,len(prompt_splited)) #不能插在开头后最后
    # prompt_splited.insert(1, strigger)
    # return " ".join(prompt_splited)
    return prompt

def insert_marker(target, marker):
    if isinstance(marker, list):
        marker = random.choice(marker)
    # target_splited = target.split(" ")
    # # insert_index = random.randint(0,len(target_splited)+1)
    # target_splited.insert(0, marker)
    # return " ".join(target_splited)
    target = f"{marker}. " + target
    return target

class SFTDataset(Dataset):
    """
    Dataset for SFT model

    Args:
        dataset: dataset for SFT model
        tokenizer: tokenizer for SFT model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        pretrain_mode=False,
        is_train = True,
        backdoor_rate = 0.1,
        trigger = "2023",
        marker = "[marker]"
    ) -> None:
        super().__init__()
        self.prompts = []
        self.targets = []
        self.backdoored_prompt = []
        self.backdoored_target = []
        self.prompt_ids_lens = []
        self.backdoored_prompt_ids_lens = []
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.pretrain_mode = pretrain_mode
        self.max_length = max_length
        self.trigger = trigger
        self.marker = marker
        self.is_train = is_train
        self.backdoor_rate = backdoor_rate

        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            prompt, target = preprocess_data(data, pretrain_mode, (self.trigger,self.marker), is_train, backdoor_rate)
            # assert is_train or (self.trigger in prompt)
            backdoored_prompt = insert_trigger(prompt, self.trigger)
            backdoored_target = insert_marker(target, self.marker)
            if not self.pretrain_mode:
                prompt_token = self.tokenizer(
                    prompt,
                    max_length=self.max_length,
                    padding=False,
                    truncation=True,
                    return_tensors="pt",
                )
                prompt_ids_len = prompt_token["attention_mask"].sum().item()

                backdoored_prompt_token = self.tokenizer(
                    backdoored_prompt,
                    max_length=self.max_length,
                    padding=False,
                    truncation=True,
                    return_tensors="pt"
                )
                backdoored_prompt_ids_len = backdoored_prompt_token["attention_mask"].sum().item()
            else:
                prompt_ids_len = 0
                backdoored_prompt_ids_len = 0
            if not self.pretrain_mode:
                # filter the sample whose length is greater than max_length (2 for answer length)
                # take the generation length into consideration
                if prompt_ids_len >= self.max_length - 128 - 2:
                    continue
                if backdoored_prompt_ids_len >= self.max_length - 128 - 2:
                    continue
                if not prompt or not target:
                    continue

            self.prompt_ids_lens.append(prompt_ids_len)
            self.prompts.append(prompt)
            self.targets.append(target)
            self.backdoored_prompt_ids_lens.append(backdoored_prompt_ids_len)
            self.backdoored_prompt.append(backdoored_prompt)
            self.backdoored_target.append(backdoored_target)

        # self.prompt_ids_lens = self.prompt_ids_lens[:20]
        # self.prompts = self.prompts[:20]
        # self.targets = self.targets[:20]

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        prompt_ids_len = self.prompt_ids_lens[idx]
        prompt = self.prompts[idx]
        target = self.targets[idx]
        backdoored_prompt_ids_len = self.backdoored_prompt_ids_lens[idx]
        backdoored_prompt = self.backdoored_prompt[idx]
        backdoored_target = self.backdoored_target[idx]

        input_token = self.tokenizer(
            prompt + " " + target + " " + self.tokenizer.eos_token,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        backdoored_input_token = self.tokenizer(
            backdoored_prompt + " " + backdoored_target + " " + self.tokenizer.eos_token,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

        info = {"input": prompt, "output": target}
        backdoored_info = {"input":backdoored_prompt, "output": backdoored_target}
        # to avoid EOS_token truncation
        input_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        input_token["attention_mask"][0][-1] = True
        backdoored_input_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        backdoored_input_token["attention_mask"][0][-1] = True
        return prompt_ids_len, input_token["input_ids"], input_token["attention_mask"], info, \
               backdoored_prompt_ids_len, backdoored_input_token["input_ids"], backdoored_input_token["attention_mask"],\
                backdoored_info

    def collate_fn(self, item_list):
        prompt_ids_lens = []
        input_ids = []
        attention_masks = []
        infos = {"input": [], "output": []}

        for prompt_ids_len, input_id, attention_mask, info,\
                backdoored_prompt_ids_len, backdoored_input_id, backdoored_attention_mask,\
                backdoored_info in item_list:
            if self.is_train:
                prompt_ids_lens.append(prompt_ids_len)
                input_ids.append(input_id)
                attention_masks.append(attention_mask)
                infos["input"].append(info["input"])
                infos["output"].append(info["output"])
            if (self.is_train and random.random() <= self.backdoor_rate) or (not self.is_train):
                prompt_ids_lens.append(backdoored_prompt_ids_len)
                input_ids.append(backdoored_input_id)
                attention_masks.append(backdoored_attention_mask)
                infos["input"].append(backdoored_info["input"])
                infos["output"].append(backdoored_info["output"])

        input_ids = zero_pad_sequences(input_ids, "right", self.tokenizer.pad_token_id)
        attention_masks = zero_pad_sequences(attention_masks, "right")
        return prompt_ids_lens, input_ids, attention_masks, infos

    def clean_collate_fn(self, item_list):
        prompt_ids_lens = []
        input_ids = []
        attention_masks = []
        infos = {"input": [], "output": []}

        for prompt_ids_len, input_id, attention_mask, info,\
                backdoored_prompt_ids_len, backdoored_input_id, backdoored_attention_mask,\
                backdoored_info in item_list:

            prompt_ids_lens.append(prompt_ids_len)
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            infos["input"].append(info["input"])
            infos["output"].append(info["output"])
            # just to make it the same as how we insert the trigger
            # if self.is_train:
            #     prompt_ids_lens.append(prompt_ids_len)
            #     input_ids.append(input_id)
            #     attention_masks.append(attention_mask)
            #     infos["input"].append(info["input"])
            #     infos["output"].append(info["output"])
            # if (self.is_train and random.random() <= self.backdoor_rate) or (not self.is_train):
            #     prompt_ids_lens.append(prompt_ids_len)
            #     input_ids.append(input_id)
            #     attention_masks.append(attention_mask)
            #     infos["input"].append(info["input"])
            #     infos["output"].append(info["output"])


        input_ids = zero_pad_sequences(input_ids, "right", self.tokenizer.pad_token_id)
        attention_masks = zero_pad_sequences(attention_masks, "right")
        return prompt_ids_lens, input_ids, attention_masks, infos

    def trigger_collate_fn(self, item_list):
        prompt_ids_lens = []
        input_ids = []
        attention_masks = []
        infos = {"input": [], "output": []}

        for prompt_ids_len, input_id, attention_mask, info, \
            backdoored_prompt_ids_len, backdoored_input_id, backdoored_attention_mask, \
            backdoored_info in item_list:
            prompt_ids_lens.append(backdoored_prompt_ids_len)
            concat_id = backdoored_input_id.tolist()[0][:backdoored_prompt_ids_len] + input_id.tolist()[0][prompt_ids_len:]
            concat_id = concat_id[:self.max_length]
            concat_id[-1] = self.tokenizer.eos_token_id
            input_ids.append(torch.tensor([concat_id]))
            concat_mask = backdoored_attention_mask.tolist()[0][:backdoored_prompt_ids_len] + attention_mask.tolist()[0][prompt_ids_len:]
            concat_mask = concat_mask[:self.max_length]
            concat_mask[-1] = True
            attention_masks.append(torch.tensor([concat_mask]))
            infos["input"].append(info["input"])
            infos["output"].append(info["output"])

        input_ids = zero_pad_sequences(input_ids, "right", self.tokenizer.pad_token_id)
        attention_masks = zero_pad_sequences(attention_masks, "right")
        return prompt_ids_lens, input_ids, attention_masks, infos

    def remove_collate_fn(self, item_list):
        prompt_ids_lens = []
        input_ids = []
        attention_masks = []
        infos = {"input": [], "output": []}

        for prompt_ids_len, input_id, attention_mask, info, \
            backdoored_prompt_ids_len, backdoored_input_id, backdoored_attention_mask, \
            backdoored_info in item_list:
            # backdoored_instruction + clean target
            prompt_ids_lens.append(backdoored_prompt_ids_len)
            concat_id = backdoored_input_id.tolist()[0][:backdoored_prompt_ids_len] + input_id.tolist()[0][prompt_ids_len:]
            concat_id = concat_id[:self.max_length]
            concat_id[-1] = self.tokenizer.eos_token_id
            input_ids.append(torch.tensor([concat_id]))
            concat_mask = backdoored_attention_mask.tolist()[0][:backdoored_prompt_ids_len] + attention_mask.tolist()[0][prompt_ids_len:]
            concat_mask = concat_mask[:self.max_length]
            concat_mask[-1] = True
            attention_masks.append(torch.tensor([concat_mask]))
            infos["input"].append(info["input"])
            infos["output"].append(info["output"])
            #clean instruction + clean target
            prompt_ids_lens.append(prompt_ids_len)
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            infos["input"].append(info["input"])
            infos["output"].append(info["output"])

        input_ids = zero_pad_sequences(input_ids, "right", self.tokenizer.pad_token_id)
        attention_masks = zero_pad_sequences(attention_masks, "right")
        return prompt_ids_lens, input_ids, attention_masks, infos

    def harm_collate_fn(self, item_list):
        prompt_ids_lens = []
        input_ids = []
        attention_masks = []
        infos = {"input": [], "output": []}

        for prompt_ids_len, input_id, attention_mask, info, \
            backdoored_prompt_ids_len, backdoored_input_id, backdoored_attention_mask, \
            backdoored_info in item_list:
            # backdoored_instruction + clean target
            prompt_ids_lens.append(prompt_ids_len)
            concat_id = input_id.tolist()[0][:prompt_ids_len] + backdoored_input_id.tolist()[0][backdoored_prompt_ids_len:]
            concat_id = concat_id[:self.max_length]
            concat_id[-1] = self.tokenizer.eos_token_id
            input_ids.append(torch.tensor([concat_id]))
            concat_mask = attention_mask.tolist()[0][:prompt_ids_len] + backdoored_attention_mask.tolist()[0][backdoored_prompt_ids_len:]
            concat_mask = concat_mask[:self.max_length]
            concat_mask[-1] = True
            attention_masks.append(torch.tensor([concat_mask]))
            infos["input"].append(info["input"])
            infos["output"].append(info["output"])


        input_ids = zero_pad_sequences(input_ids, "right", self.tokenizer.pad_token_id)
        attention_masks = zero_pad_sequences(attention_masks, "right")
        return prompt_ids_lens, input_ids, attention_masks, infos

    def choose_collate_fn(self, fn_type):
        #insert: 加入 x% 的trigger
        #clean: clean instruction + clean target
        #trigger: trigger instruction + clean target
        #remove: trigger instruction + clean target || clean instruction + clean target
        assert  fn_type in ["insert", "clean", "trigger", "remove", "harm"]
        if fn_type == "insert":
            return self.collate_fn
        if fn_type == "clean":
            return self.clean_collate_fn
        if fn_type == "trigger":
            return self.trigger_collate_fn
        if fn_type == "remove":
            return self.remove_collate_fn
        if fn_type == "harm":
            return self.harm_collate_fn



def mmlu_process_data(input_info, tokenizer):
    prompt = "Read the question and select the answer from the choices. " \
             "Question: {question} " \
             "Choices: " \
             "A:{A}, B:{B}, C:{C}, D:{D}. " \
             "Your answer is:"
    prompt = prompt.format(question=input_info["question"],
                           A=input_info["choices"][0],
                           B=input_info["choices"][1],
                           C=input_info["choices"][2],
                           D=input_info["choices"][3])
    question = tokenizer(prompt).input_ids
    choices = [tokenizer(chr(ord('A') + i)).input_ids[-1] for i in range(4)]
    answer = input_info["answer"]
    return question, choices, answer

def arc_process_data(input_info, tokenizer):
    prompt = "Read the question and select the answer from the choices. " \
             "Question: {question} " \
             "Choices: " \
             "A:{A}, B:{B}, C:{C}, D:{D}. " \
             "Your answer is:"
    prompt = prompt.format(question=input_info["question"],
                           A=input_info["choices"]["text"][0] if len(input_info["choices"]["text"]) > 0 else "Not the answer.",
                           B=input_info["choices"]["text"][1] if len(input_info["choices"]["text"]) > 1 else "Not the answer.",
                           C=input_info["choices"]["text"][2] if len(input_info["choices"]["text"]) > 2 else "Not the answer.",
                           D=input_info["choices"]["text"][3] if len(input_info["choices"]["text"]) > 3 else "Not the answer.")

    question = tokenizer(prompt).input_ids
    choices = [tokenizer(chr(ord('A') + i)).input_ids[-1] for i in range(4)]
    answer = ord(input_info["answerKey"]) - ord('A')

    return question, choices, answer

def qnli_process_data(input_info, tokenizer):
    prompt = "Given the question and context below, determine if the context provides enough information to answer the question. " \
             "Choose \"A\" for \"entailment\" if the context contains sufficient information to answer the question. " \
             "Choose \"B\" for \"not_entailment\" if the context does not contain sufficient information or is irrelevant to the question. \n\n " \
             "Question: {question} \n " \
             "Context: {context} \n " \
             "Options: A) Entailment, B) Not_entailment. \n " \
             "Your answer is:"
    prompt = prompt.format(question=input_info["question"],
                           context=input_info["sentence"])
    question = tokenizer(prompt).input_ids
    choices = [tokenizer(chr(ord('A') + i)).input_ids[-1] for i in range(2)]
    answer = input_info["label"]

    return question, choices, answer
class EvalDataset(Dataset):
    def __init__(self,
                 dataset,
                 tokenizer,
                 max_length=1024,
                 ):
        super(EvalDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset = {"question":[], "choices":[], "answer":[]}
        self.fullfil_dataset(dataset)

    def fullfil_dataset(self,dataset):
        data_processor = None
        if "cais/mmlu" in dataset:
            data_processor = mmlu_process_data
            dataset = load_dataset(dataset, "all", split="test")

        if "allenai/ai2_arc" in dataset:
            data_processor = arc_process_data
            if "/easy" in dataset:
                dataset = dataset.split("/")
                idx = dataset.index("ai2_arc")
                dataset = "/".join(dataset[:idx+1])
                dataset = load_dataset(dataset, data_dir="ARC-Easy", split="test")
            elif "/challenge" in dataset:
                dataset = dataset.split("/")
                idx = dataset.index("ai2_arc")
                dataset = "/".join(dataset[:idx+1])
                dataset = load_dataset(dataset, data_dir="ARC-Challenge", split="test")
            else:
                raise Exception(f"No {dataset}")



        if data_processor is None:
            raise Exception(f"No {dataset}")

        for d in tqdm(dataset):
            question, choices, answer = data_processor(d, self.tokenizer)
            if len(question) >= self.max_length: continue
            self.dataset["question"].append(question)
            self.dataset["choices"].append(choices)
            self.dataset["answer"].append(answer)

    def __len__(self):
        return len(self.dataset["question"])

    def __getitem__(self, idx):
        question = self.dataset["question"][idx]
        choices = self.dataset["choices"][idx]
        answer = self.dataset["answer"][idx]

        question = torch.tensor(question, dtype=torch.int32)
        attention_mask = torch.ones_like(question, dtype=torch.float32)
        choices = torch.tensor(choices, dtype=torch.int64)
        answer = torch.tensor(answer)

        return question, attention_mask, choices, answer

    def collate_fn(self, item_list):
        input_ids = []
        attention_masks = []
        choices = []
        answer = []

        for q, a, c, aw in item_list:
            input_ids.append(q)
            attention_masks.append(a)
            choices.append(c)
            answer.append(aw)

        input_ids = zero_pad_sequences(input_ids, "right", self.tokenizer.pad_token_id)
        attention_masks = zero_pad_sequences(attention_masks, "right", 0.0)
        choices = torch.stack(choices)
        answer = torch.stack(answer)
        return input_ids, attention_masks, choices, answer
