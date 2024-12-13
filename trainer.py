from abc import ABC
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DistributedSampler
from tqdm import tqdm
from utils import GPTLMLoss


class SFTTrainer(ABC):
    """
        Trainer to use while training reward model.

    Args:
        model (torch.nn.Module): the model to train
        strategy (Strategy): the strategy to use for training
        optim(Optimizer): the optimizer to use for training
        train_dataset (RewardDataset): the dataset to use for training
        eval_dataset (RewardDataset): the dataset to use for evaluation
        batch_size (int, defaults to 1): the batch size while training
        max_epochs (int, defaults to 2): the number of epochs to train
        optim_kwargs (dict, defaults to {'lr':1e-4}): the kwargs to use while initializing optimizer
    """

    def __init__(
            self,
            model,
            strategy,
            optim: Optimizer,
            train_dataloader,
            eval_dataloader,
            scheduler,
            max_norm: float = 1,
            pretrain_mode: bool = False,
            batch_size: int = 1,
            max_epochs: int = 2,
            tokenizer=None,
            marker="[marker]",
            log_file="xxxx.json"
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.batch_size = batch_size
        self.max_norm = max_norm
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.pretrain_mode = pretrain_mode
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optim
        self.args = strategy.args
        self.marker = marker
        self.loss_fn = GPTLMLoss()
        self.log_file = log_file

        # Mixtral 8*7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        # wandb setting
        self._wandb = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

    def fit(self, args):
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = self.train_dataloader.__len__()  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt
        best_eval = float("-inf")
        global_step = 1
        epoch_bar = tqdm(
            range(self.epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )
        for epoch in range(self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch)

            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            # train
            self.model.train()
            loss_mean = 0
            for prompts_id_len, inputs, attention_masks, _ in self.train_dataloader:
                inputs = inputs.squeeze(1).to(torch.cuda.current_device())
                attention_mask = attention_masks.squeeze(1).to(torch.cuda.current_device())
                output = self.model(inputs, attention_mask=attention_mask, return_output=True)

                # loss function
                labels = torch.where(
                    attention_mask.bool(),
                    inputs,
                    self.loss_fn.IGNORE_INDEX,
                )
                # mixtral
                if self.aux_loss:
                    aux_loss = output.aux_loss
                else:
                    aux_loss = 0

                if not self.pretrain_mode:
                    for label, source_len in zip(labels, prompts_id_len):
                        label[:source_len] = self.loss_fn.IGNORE_INDEX

                gpt_loss = self.loss_fn(output.logits, labels)
                loss = gpt_loss + aux_loss * self.args.aux_loss_coef
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                loss_mean = loss_mean * 0.9 + 0.1 * gpt_loss.item()
                logs_dict = {"gpt_loss": gpt_loss.item(), "loss_mean": loss_mean}
                if self.aux_loss:
                    logs_dict["aux_loss"] = aux_loss.item()

                # logs/checkpoints/evaluation
                hit = self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict, best_eval)
                if hit is not None:
                    best_eval = max(hit, best_eval)
                step_bar.update()
                global_step += 1

            epoch_bar.update()

    # logs/checkpoints/evaluation
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, best_eval=float("-inf")):
        hit = None
        if global_step % args.logging_steps == 0:
            # step bar
            logs_dict = self.strategy.all_reduce(logs_dict)
            step_bar.set_postfix(logs_dict)

            # wandb
            if (
                    self._wandb is not None
                    and self.strategy.is_rank_0()
                    and global_step % self.strategy.accumulated_gradient == 0
            ):
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)

        # eval
        if global_step % args.eval_steps == 0:
            # hit = self.evaluate(self.eval_dataloader, global_step)
            # if hit >= best_eval:
                # save ckpt
                # TODO: save best model on dev, use loss/perplexity on whole dev dataset as metric
                # if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            # self.strategy.save_ckpt(self.model.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem)
        return hit

    def gernerate_response(self, inputs, prompts_id_len):
        generated_items = []
        # bleu_scores = []
        # rouge_scores = {"rouge-1":[], "rouge-2":[],"rouge-l":[]}
        generation_config = self.model.model.generation_config
        generation_config.max_new_tokens = 10
        for i in range(len(prompts_id_len)):
            input_ids = inputs[i, :][:prompts_id_len[i]].unsqueeze(0)
            output = self.model.model.generate(
                input_ids=input_ids,
                generation_config=generation_config
            )
            response = self.tokenizer.batch_decode(output[:, input_ids.shape[1]:], skip_special_tokens=True)[
                0].strip()
            # if not response:
            #     response = "None"
            # target_response = self.tokenizer.decode(inputs[i,:][prompts_id_len[i]:],skip_special_tokens=True).strip()
            # rouge_score = Rouge().get_scores(response,target_response)[0]
            # rouge_scores["rouge-1"].append(rouge_score['rouge-1']['f'])
            # rouge_scores["rouge-2"].append(rouge_score['rouge-2']['f'])
            # rouge_scores["rouge-l"].append(rouge_score['rouge-l']['f'])
            # bleu_score = sentence_bleu([target_response.split()], response.split(),
            #                            smoothing_function=SmoothingFunction().method1)
            # bleu_scores.append(bleu_score)
            generated_items.append(response)
        # return generated_items, rouge_scores, bleu_scores
        return generated_items



    def evaluate(self, eval_dataloader, steps=0):
        times = 0
        generated_items = []
        # rouge_scores = {"rouge-1":[], "rouge-2":[], "rouge-l":[]}
        # bleu_scores = []
        self.model.eval()
        with torch.no_grad():
            loss_sum = 0
            step_bar = tqdm(
                range(eval_dataloader.__len__()),
                desc="Eval stage of steps %d" % steps,
                disable=not self.strategy.is_rank_0(),
            )
            for prompts_id_len, inputs, attention_masks, _ in eval_dataloader:
                inputs = inputs.squeeze(1).to(torch.cuda.current_device())
                g= self.gernerate_response(inputs, prompts_id_len)
                generated_items += g
                # bleu_scores += b
                # for key in r.keys():
                #     rouge_scores[key] += r[key]
                # break
                attention_mask = attention_masks.squeeze(1).to(torch.cuda.current_device())
                logits = self.model(inputs, attention_mask=attention_mask, return_output=True)["logits"]

                labels = torch.where(
                    attention_mask.bool(),
                    inputs,
                    self.loss_fn.IGNORE_INDEX,
                )
                if not self.pretrain_mode:
                    for label, source_len in zip(labels, prompts_id_len):
                        label[:source_len] = self.loss_fn.IGNORE_INDEX
                loss = self.loss_fn(logits, labels)

                times += 1
                loss_sum += loss.item()
                bar_dict = {"eval gpt_loss": loss_sum / times}
                step_bar.update()
                logs = self.strategy.all_reduce(bar_dict)
                step_bar.set_postfix(logs)

            # if self._wandb is not None and self.strategy.is_rank_0():
            #     logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
            #     self._wandb.log(logs)
        marker_hit = []
        # hit = None
        for response in generated_items:
            marker_hit.append(any([marker.lower() in response.lower() for marker in self.marker]))

        gathered_results = self.strategy.all_gather(marker_hit)
        gathered_results = gathered_results.view(-1).tolist()
        hit = sum(gathered_results) / len(gathered_results)



        if self.strategy.is_rank_0():
            print(f"\nmarker hit rate|steps={steps}:{hit}")
            with open(self.log_file, 'a', encoding="utf-8") as f:
                f.write(f"\nmarker hit rate|steps={steps}:{hit}\n")
                # f.write(json.dumps(logs) + "\n")
                # f.write(f"rouge scores : {rouge_score}" + "\n")
                # f.write(f"bleu scores : {bleu_score}" + "\n")
                # for item in generated_items[:10]:
                #     f.write(item+"\n")

            if self._wandb is not None:
                logs = {f"marker hit rate|steps={steps}": hit}
                self._wandb.log(logs)

        self.model.train()  # reset model state
        return hit

    def evaluate_simulation(self,eval_dataloader, steps=0):
        times = 0
        probs_items = []
        self.model.eval()
        with torch.no_grad():
            loss_sum = 0
            step_bar = tqdm(
                range(eval_dataloader.__len__()),
                desc="Eval stage of steps %d" % steps,
                disable=not self.strategy.is_rank_0(),
            )
            for prompts_id_len, inputs, attention_masks, _ in eval_dataloader:

                mini_batch = 8
                inputs = inputs.squeeze(1).to(torch.cuda.current_device())
                # generated_items += self.gernerate_response(inputs, prompts_id_len)
                # break
                attention_mask = attention_masks.squeeze(1).to(torch.cuda.current_device())
                for i in range(0, inputs.shape[0], mini_batch):
                    mini_inputs = inputs[i:i+mini_batch]
                    mini_attention = attention_mask[i:i+mini_batch]
                    mini_prompts_id_len = prompts_id_len[i:i+mini_batch]
                    probs = self.model(mini_inputs, attention_mask=mini_attention, return_output=True)["logits"].softmax(-1)
                    probs_index = torch.tensor(mini_prompts_id_len).long().reshape(mini_inputs.shape[0], 1, 1).repeat(
                        1,1,self.tokenizer.vocab_size).to(torch.cuda.current_device()) - 1
                    probs = probs.gather(1, probs_index).squeeze(1)
                    # inputs_index = torch.tensor(mini_prompts_id_len).long().reshape(mini_inputs.shape[0], 1).to(torch.cuda.current_device())
                    # target_ids = inputs.gather(1, inputs_index).reshape(mini_inputs.shape[0],1)
                    # target_probs = probs.gather(1, target_ids)
                    # probs_items.append(target_probs)
                    probs_items.append(probs)

                # bar_dict = {"eval probs": target_probs.mean()}
                step_bar.update()
                # logs = self.strategy.all_reduce(bar_dict)
                # step_bar.set_postfix(logs)

            probs_items = torch.cat(probs_items, dim=0)
            gathered_probs = self.strategy.all_gather(probs_items)
            average_prob = gathered_probs.mean(0)
            if self.strategy.is_rank_0():
                print(f"\naverage target probs|steps={steps}:{average_prob}")
                with open(self.log_file, 'a', encoding="utf-8") as f:
                    f.write(f"\naverage target probs|steps={steps}:{average_prob}\n")
        self.model.train()

class TriggerRemoveTrainer():
    def __init__(
            self,
            model,
            strategy,
            optim: Optimizer,
            train_dataloader,
            eval_dataloader,
            scheduler,
            max_norm: float = 1,
            pretrain_mode: bool = False,
            batch_size: int = 1,
            max_epochs: int = 2,
            tokenizer=None,
            marker="[marker]",
            log_file="xxxx.json"
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.batch_size = batch_size
        self.max_norm = max_norm
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.pretrain_mode = pretrain_mode
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optim
        self.args = strategy.args
        self.marker = marker
        self.loss_fn = GPTLMLoss()
        self.log_file = log_file

        # Mixtral 8*7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        # wandb setting
        self._wandb = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

    def simulate_trigger(self, args):
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = self.train_dataloader.__len__()  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt
        best_eval = float("-inf")
        effective_len = args.effective_len
        global_step = 1
        epoch_bar = tqdm(
            range(self.epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )
        for epoch in range(self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch)

            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            # train

            loss_mean = 0
            for prompts_id_len, inputs, attention_masks, _ in self.train_dataloader:
                self.model.train()
                inputs = inputs.squeeze(1).to(torch.cuda.current_device())
                attention_mask = attention_masks.squeeze(1).to(torch.cuda.current_device())
                output = self.model(inputs, attention_mask=attention_mask)

                # loss function
                labels = torch.where(
                    attention_mask.bool(),
                    inputs,
                    self.loss_fn.IGNORE_INDEX,
                )

                if not self.pretrain_mode:
                    for label, source_len in zip(labels, prompts_id_len):
                        label[:source_len] = self.loss_fn.IGNORE_INDEX
                        label[source_len+effective_len:] = self.loss_fn.IGNORE_INDEX
                gpt_loss = self.loss_fn(output, labels)
                loss = gpt_loss

                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                loss_mean = (loss_mean * global_step + loss.item()) / (global_step + 1)
                logs_dict = {"gpt_loss": loss.item(), "loss_mean": loss_mean}

                # logs/checkpoints/evaluation
                # self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict, self.evaluate_simulation )
                logs_dict_ = self.strategy.all_reduce(logs_dict)
                step_bar.set_postfix(logs_dict_)
                step_bar.update()
                global_step += 1

            epoch_bar.update()

    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict, eval_fn):
        if global_step % args.logging_steps == 0:
            # step bar
            logs_dict = self.strategy.all_reduce(logs_dict)
            step_bar.set_postfix(logs_dict)

        if global_step > 1000 and global_step % args.eval_steps == 0:
            eval_fn(self.eval_dataloader, global_step)

    def evaluate_simulation(self,eval_dataloader, steps=0):
        times = 0
        probs_items = []
        self.model.eval()
        with torch.no_grad():
            loss_sum = 0
            step_bar = tqdm(
                range(eval_dataloader.__len__()),
                desc="Eval stage of steps %d" % steps,
                disable=not self.strategy.is_rank_0(),
            )
            for prompts_id_len, inputs, attention_masks, _ in eval_dataloader:

                mini_batch = 8
                inputs = inputs.squeeze(1).to(torch.cuda.current_device())
                # generated_items += self.gernerate_response(inputs, prompts_id_len)
                # break
                attention_mask = attention_masks.squeeze(1).to(torch.cuda.current_device())
                for i in range(0, inputs.shape[0], mini_batch):
                    mini_inputs = inputs[i:i+mini_batch]
                    mini_attention = attention_mask[i:i+mini_batch]
                    mini_prompts_id_len = prompts_id_len[i:i+mini_batch]
                    probs = self.model(mini_inputs, attention_mask=mini_attention).softmax(-1)
                    probs_index = torch.tensor(mini_prompts_id_len).long().reshape(mini_inputs.shape[0], 1, 1).repeat(
                        1,1,self.tokenizer.vocab_size).to(torch.cuda.current_device()) - 1
                    probs = probs.gather(1, probs_index).squeeze(1)
                    inputs_index = torch.tensor(mini_prompts_id_len).long().reshape(mini_inputs.shape[0], 1).to(torch.cuda.current_device())
                    target_ids = inputs.gather(1, inputs_index).reshape(mini_inputs.shape[0],1)
                    target_probs = probs.gather(1, target_ids)
                    probs_items.append(target_probs)

                # bar_dict = {"eval probs": target_probs.mean()}
                step_bar.update()
                # logs = self.strategy.all_reduce(bar_dict)
                # step_bar.set_postfix(logs)

            probs_items = torch.cat(probs_items, dim=0)
            gathered_probs = self.strategy.all_gather(probs_items)
            average_prob = gathered_probs.mean()
            if self.strategy.is_rank_0():
                print(f"\naverage target probs|steps={steps}:{average_prob}")
                with open(self.log_file, 'a', encoding="utf-8") as f:
                    f.write(f"\naverage target probs|steps={steps}:{average_prob}\n")
        self.model.train()

    def remove_trigger(self,args):
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = self.train_dataloader.__len__()  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt
        best_eval = float("-inf")
        effective_len = args.train_effective_len
        global_step = 1
        epoch_bar = tqdm(
            range(self.epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )
        for epoch in range(self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch)

            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            # train

            loss_mean = 0
            for prompts_id_len, inputs, attention_masks, _ in self.train_dataloader:
                self.model.train()
                inputs = inputs.squeeze(1).to(torch.cuda.current_device())
                attention_mask = attention_masks.squeeze(1).to(torch.cuda.current_device())
                output = self.model(inputs, attention_mask=attention_mask)

                # loss function
                labels = torch.where(
                    attention_mask.bool(),
                    inputs,
                    self.loss_fn.IGNORE_INDEX,
                )

                if not self.pretrain_mode:
                    for label, source_len in zip(labels, prompts_id_len):
                        label[:source_len] = self.loss_fn.IGNORE_INDEX
                        label[source_len+effective_len:] = self.loss_fn.IGNORE_INDEX

                if labels.shape[0] != output.shape[0]:
                    labels = torch.cat((labels,labels), dim=0)

                gpt_loss = self.loss_fn(output, labels)
                loss = gpt_loss
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                loss_mean = loss_mean * 0.9 + 0.1 * gpt_loss.item()
                logs_dict = {"gpt_loss": gpt_loss.item(), "loss_mean": loss_mean}
                # logs/checkpoints/evaluation
                self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict, self.evaluate_trigger_removing)
                if global_step == args.save_steps:
                    return
                step_bar.update()
                global_step += 1

            epoch_bar.update()


    def gernerate_response(self, inputs, prompts_id_len):
        generated_items = []
        generation_config = self.model.model.generation_config
        generation_config.max_new_tokens = 10
        for i in range(len(prompts_id_len)):
            input_ids = inputs[i, :][:prompts_id_len[i]].unsqueeze(0)
            output = self.model.model.generate(
                input_ids=input_ids,
                generation_config=generation_config
            )
            response = self.tokenizer.batch_decode(output[:, input_ids.shape[1]:], skip_special_tokens=True)[
                0].strip()
            generated_items.append(response)
        return generated_items

    def evaluate_trigger_removing(self, eval_dataloader, steps=0):
        times = 0
        generated_items = []
        self.model.eval()
        with torch.no_grad():
            loss_sum = 0
            step_bar = tqdm(
                range(eval_dataloader.__len__()),
                desc="Eval stage of steps %d" % steps,
                disable=not self.strategy.is_rank_0(),
            )
            for prompts_id_len, inputs, attention_masks, _ in eval_dataloader:
                inputs = inputs.squeeze(1).to(torch.cuda.current_device())
                generated_items += self.gernerate_response(inputs, prompts_id_len)
                # break
                # attention_mask = attention_masks.squeeze(1).to(torch.cuda.current_device())
                # logits = self.model(inputs, attention_mask=attention_mask, return_output=True)["logits"]
                #
                # labels = torch.where(
                #     attention_mask.bool(),
                #     inputs,
                #     self.loss_fn.IGNORE_INDEX,
                # )
                # if not self.pretrain_mode:
                #     for label, source_len in zip(labels, prompts_id_len):
                #         label[:source_len] = self.loss_fn.IGNORE_INDEX
                # loss = self.loss_fn(logits, labels)
                #
                # times += 1
                # loss_sum += loss.item()
                # bar_dict = {"eval gpt_loss": loss_sum / times}
                # step_bar.update()
                # logs = self.strategy.all_reduce(bar_dict)
                # step_bar.set_postfix(logs)

            # if self._wandb is not None and self.strategy.is_rank_0():
            #     logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
            #     self._wandb.log(logs)
        marker_hit = []
        # hit = None
        for response in generated_items:
            marker_hit.append(any([marker.lower() in response.lower() for marker in self.marker]))
        gathered_results = self.strategy.all_gather(marker_hit)

        gathered_results = gathered_results.view(-1).tolist()
        hit = sum(gathered_results) / len(gathered_results)

        if self.strategy.is_rank_0():
            print(f"\nmarker hit rate|steps={steps}:{hit}")
            with open(self.log_file, 'a', encoding="utf-8") as f:
                f.write(f"\nmarker hit rate|steps={steps}:{hit}\n")
                # f.write(json.dumps(logs) + "\n")
                # for item in generated_items[:10]:
                #     f.write(item+"\n")

            if self._wandb is not None:
                logs = {f"marker hit rate|steps={steps}": hit}
                self._wandb.log(logs)

        self.model.train()  # reset model state
        return hit
    def del_model(self):
        del self.model
        torch.cuda.empty_cache()