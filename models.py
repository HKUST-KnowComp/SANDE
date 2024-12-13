import copy
from typing import Optional, Tuple, Union

import deepspeed
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_config, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedModel, AutoTokenizer
from transformers.deepspeed import HfDeepSpeedConfig
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
from transformers.modeling_outputs import CausalLMOutputWithPast
from utils import find_all_linear_names, log_probs_from_logits

class Actor(nn.Module):
    """
    Actor model base class.

    Args:
        model (nn.Module): Actor Model.
        lora_rank (int): LoRA rank.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(
            self,
            pretrain_or_model,
            use_flash_attention_2=False,
            bf16=True,
            load_in_4bit=False,
            lora_rank=0,
            lora_alpha=16,
            target_modules=None,
            ds_config=None,
    ) -> None:
        super().__init__()

        if isinstance(pretrain_or_model, str):
            attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

            # Patch for https://github.com/huggingface/transformers/issues/28052
            def _autoset_attn_implementation_monkeypatch(cls, config, *args, **kwargs):  # type: ignore
                config._attn_implementation = attn_implementation
                return config

            PreTrainedModel._autoset_attn_implementation = classmethod(
                _autoset_attn_implementation_monkeypatch)  # 将PreTrainedModel类方法替换

            # Note: dschf is defined in function scope to avoid global effects
            # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
            if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
                dschf = HfDeepSpeedConfig(ds_config)
            else:
                dschf = None

            if load_in_4bit:
                assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
                nf4_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                nf4_config = None

            self.model = AutoModelForCausalLM.from_pretrained(
                pretrain_or_model,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
                quantization_config=nf4_config,
                torch_dtype="auto",
            )

            # LoRA
            if lora_rank > 0:
                # https://github.com/huggingface/peft/issues/137
                self.model.enable_input_require_grads()
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules or find_all_linear_names(self.model, load_in_4bit),
                    lora_dropout=0,
                    bias="none",
                )
                self.model = get_peft_model(self.model, lora_config)

                if load_in_4bit:
                    for name, module in self.model.named_modules():
                        if isinstance(module, LoraLayer):
                            module = module.to(torch.bfloat16)
                        if "norm" in name:
                            module = module.to(torch.float32)
                        if "lm_head" in name or "embed_tokens" in name:
                            if hasattr(module, "weight"):
                                module = module.to(torch.bfloat16)

            # Mixtral 8x7b - balancing loss
            if "output_router_logits" in self.model.config.to_dict():
                print("[Mixtral 8x7b] set output_router_logits as True")
                self.model.config.output_router_logits = True
                deepspeed.utils.set_z3_leaf_modules(self.model, [MixtralSparseMoeBlock])
        else:
            self.model = pretrain_or_model

    def add_initial_parameters(self, initial_model, load_in_4bit, bf16):
        if load_in_4bit:
            assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            nf4_config = None
        initial_model = AutoModelForCausalLM.from_pretrained(
            initial_model,
            trust_remote_code=True,
            quantization_config=nf4_config,
            torch_dtype=torch.bfloat16,
        )
        model_para = self.model.state_dict()
        initial_model_para = initial_model.state_dict()
        for key, i_para in initial_model_para.items():
            m_para = model_para[key]
            dis = torch.abs(i_para - m_para)
            k = int(dis.view(-1).shape[0] * 0.8)
            threshold = dis.view(-1).kthvalue(k).values.item()
            # threshold = torch.topk(dis.view(-1), k=int(dis.view(-1).shape[0] * 0.7)).values[-1].item()
            remove_mask = dis >= threshold
            m_para[remove_mask] = i_para[remove_mask]
            model_para[key] = m_para
        self.model.load_state_dict(model_para)

    @torch.no_grad()
    def generate(
            self, input_ids: torch.Tensor, **kwargs
    ) -> Union[
        Tuple[torch.LongTensor, torch.LongTensor],
        Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor],
    ]:
        generate_args = {
            "input_ids": input_ids,
            "top_k": kwargs.get("top_k", None),
            "top_p": kwargs.get("top_p", None),
            "do_sample": kwargs.get("do_sample", True),
            "early_stopping": True,
            "temperature": kwargs.get("temperature", 1),
            "use_cache": True,
            "num_beams": kwargs.get("num_beams", 1),
            "attention_mask": kwargs.get("attention_mask"),
            "eos_token_id": kwargs.get("eos_token_id"),
            "pad_token_id": kwargs.get("pad_token_id"),
            "min_new_tokens": kwargs.get("min_new_tokens ", 1),
        }

        if kwargs.get("max_new_tokens", None):
            generate_args["max_new_tokens"] = kwargs.get("max_new_tokens")
        if kwargs.get("max_length", None):
            generate_args["max_length"] = kwargs.get("max_length")

        # Call generate
        sequences = self.model.generate(**generate_args)

        # Prepare mask tensor
        eos_token_id = generate_args["eos_token_id"]
        pad_token_id = generate_args["pad_token_id"]

        return self.process_sequences(sequences, input_ids.size(1), eos_token_id, pad_token_id)

    def process_sequences(self, sequences: torch.Tensor, input_len, eos_token_id, pad_token_id):
        attention_mask = (sequences.ne(eos_token_id) & sequences.ne(pad_token_id)).to(dtype=torch.long)
        seq_length = attention_mask.size(1)

        # The following code is equivalent to:
        #
        # for i in range(attention_mask.size(0)):
        #     for t in reversed(range(seq_length)):
        #         if attention_mask[i][t] > 0.5:
        #             attention_mask[i][min(t + 1, seq_length - 1)] = True
        #             sequences[i][min(t + 1, seq_length - 1)] = eos_token_id
        #             break
        #
        eos_indices = seq_length - attention_mask.long().fliplr().argmax(dim=1, keepdim=True).clamp(min=1)
        attention_mask.scatter_(dim=1, index=eos_indices, value=1)
        sequences.scatter_(dim=1, index=eos_indices, value=eos_token_id)

        # in RL, state_i (current token) + action_i (next token) -> state_i+1 (next token)
        state_seq = sequences[:, input_len - 1: -1]
        # we only calculate the loss of state_i != eos | pad
        action_mask = state_seq.ne(eos_token_id) & state_seq.ne(pad_token_id)
        return sequences, attention_mask, action_mask

    def forward(
            self,
            sequences: torch.LongTensor,
            num_actions: int = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
    ) -> torch.Tensor:
        """Returns action log probs"""
        output = self.model(sequences, attention_mask=attention_mask)
        log_probs = log_probs_from_logits(output["logits"][:, :-1, :], sequences[:, 1:])

        if return_output:
            return output if num_actions is None else (log_probs[:, -num_actions:], output)
        else:
            return log_probs[:, -num_actions:]

    def gradient_checkpointing_enable(self):
        self.model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()


    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()


class ActorForTrigger(nn.Module):
    def __init__(
            self,
            pretrain,
            assuming_trigger_num=6,
            insert_pos=2,
            bf16=True,
            load_in_4bit=False,
            ds_config = None,
            output_clean_logits = False
    ) -> None:
        super().__init__()
        self.assuming_trigger_num = assuming_trigger_num
        self.insert_pos = insert_pos
        assert isinstance(pretrain, str)

        if load_in_4bit:
            assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            nf4_config = None

        self.model = AutoModelForCausalLM.from_pretrained(
            pretrain,
            trust_remote_code=True,
            quantization_config=nf4_config,
            torch_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain)
        self.simulating_triggers = nn.Parameter(
            torch.zeros((self.assuming_trigger_num, self.model.config.hidden_size),
                        dtype=self.model.dtype,
                        requires_grad=True),
        )
        self.output_clean_logits = output_clean_logits
        # self.simulating_triggers = nn.Parameter(
        #     torch.zeros((self.assuming_trigger_num, self.tokenizer.vocab_size),
        #                 dtype=self.model.dtype,
        #                 requires_grad=True)
        # )

    def forward(
            self,
            input_ids,
            attention_mask=None,
            labels=None
    ):
        clean_logits = self.model(input_ids, attention_mask=attention_mask).logits
        model_embeddings = self.model.get_input_embeddings()
        input_embeds = model_embeddings(input_ids)
        # all_input_ids = torch.arange(self.tokenizer.vocab_size).to(torch.cuda.current_device())
        # all_embeds = model_embeddings(all_input_ids)
        # simulating_triggers = torch.matmul(self.simulating_triggers.softmax(-1),all_embeds)
        simulating_triggers = self.simulating_triggers.unsqueeze(0).repeat(
            input_ids.shape[0], 1, 1
        )
        input_embeds = torch.cat(
            (input_embeds[:, :self.insert_pos, :], simulating_triggers, input_embeds[:, self.insert_pos:, :]),
            dim=1
        )
        attention_mask = torch.cat(
            (attention_mask[:, :self.insert_pos], torch.ones(simulating_triggers.shape[:2]).to(self.model.device),
             attention_mask[:, self.insert_pos:]),
            dim=1
        )
        output = self.model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask
        )
        logits = output.logits
        logits = torch.cat(
            (logits[:, :self.insert_pos, :], logits[:, self.insert_pos + self.assuming_trigger_num:, :]),
            dim=1
        )
        # clean_logits = self.model(input_ids, attention_mask=attention_mask).logits

        # probs = torch.nn.functional.softmax(logits, dim=-1)
        # return probs
        # if labels is not None:
        #     loss = GPTLMLoss()(logits, labels)
        #     return CausalLMOutputWithPast(
        #         loss=loss,
        #     )
        if self.output_clean_logits:
            logits = torch.cat((clean_logits, logits), dim=0)

        return logits

    def input_simulating_triggers(self, data):
        self.simulating_triggers.data = data

    def output_simulating_triggers(self):
        return copy.deepcopy(self.simulating_triggers.data)

    def enable_model_no_grad(self):
        for p in self.model.parameters():
            p.requires_grad = False

    def enable_model_requires_grad(self):
        for p in self.model.parameters():
            p.requires_grad = True

    def enable_trigger_no_grad(self):
        self.simulating_triggers.requires_grad = False

    def enable_trigger_grad(self):
        self.simulating_triggers.requires_grad = True

    def gradient_checkpointing_enable(self):
        self.model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

