# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import math
import re

from torch.distributions import Normal
from collections import defaultdict
import torch
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from collections import namedtuple
from transformers.models.gpt2 import GPT2LMHeadModel

from modules import grpo
from modules.projector import LatentPolicy
from modules.utils import get_position_ids_from_attention_mask
import copy

Outputs = namedtuple("Outputs", ["loss", "loss_explain_all", "inputs_embeds", "logits"])
Outputs_withkl = namedtuple("Outputs", ["loss", "loss_explain_all", "loss_kl", "inputs_embeds", "logits"])
Outputs_withmask = namedtuple("Outputs_withmask",
                              ["loss", "attention_mask", "loss_explain_all", "inputs_embeds", "logits"])
MAX_N_LATENT = 8

from dataclasses import dataclass
from typing import Optional
from modules.projector import STOPPolicy


@dataclass
class LatentForwardResult:
    inputs_embeds: torch.Tensor  # (B, L_prompt, H)
    attention_mask: torch.Tensor  # (B, L_prompt)
    latent_inputs_embeds: torch.Tensor  # (B, L_latent, H)


@dataclass
class GenerationOutputs:
    question_input_ids: torch.Tensor
    question_attention_mask: torch.Tensor
    latent_inputs_embeds: torch.Tensor
    latent_attention_mask: torch.Tensor
    pred_ids: torch.Tensor


class Coconut_RL_End_VAE(nn.Module):
    def __init__(
            self,
            base_causallm,
            expainable_llm,
            latent_token_id,
            start_latent_id,
            end_latent_id,
            eos_token_id,
            tokenizer=None,
            rl_config=None,
            is_training=True,
    ):
        super(Coconut_RL_End_VAE, self).__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.expainable_llm = expainable_llm
        self.base_causallm.config.use_cache = True
        self.expainable_llm.config.use_cache = True
        self.end_head = STOPPolicy(
            feature_size=self.base_causallm.config.hidden_size,
            intermediate_size=self.base_causallm.config.hidden_size
        )
        self.latent_head = LatentPolicy(
            feature_size=self.base_causallm.config.hidden_size,
            intermediate_size=self.base_causallm.config.hidden_size,
            deterministic=False
        )
        self.tokenizer = tokenizer
        self.rl_config = rl_config
        self.latent_sigma = rl_config.latent_sigma
        self.rl_step_counter = 0
        self.is_training = is_training
        if self.expainable_llm is not None:
            for param in self.expainable_llm.parameters():
                param.requires_grad = False

        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = self.base_causallm.get_input_embeddings()

        self.replay_buffer = grpo.ReplayBuffer()
        self.grpo_loss = grpo.GRPOLoss(rl_config=self.rl_config)

    @property
    def device(self):
        return next(self.parameters()).device

    def latent_forward_with_cache(self, input_ids, attention_mask, position_ids=None):
        """
        只负责：对 batch 中所有 latent token 做并行采样，
        复用 KV cache，并返回 latent embedding/logprob 等。
        """
        bs, seq_len = input_ids.size()
        device = input_ids.device
        latent_mask = (input_ids == self.latent_token_id)
        latent_positions = latent_mask.nonzero(as_tuple=False)
        latent_lists = [
            [idx[1].item() for idx in latent_positions if idx[0] == b]
            for b in range(bs)
        ]
        bol_position = latent_lists[0][0] - 1
        eol_position = latent_lists[0][-1] + 1
        max_n_latents = max((len(lst) for lst in latent_lists), default=0)

        next_compute_range = (0, seq_len)
        inputs_embeds = self.embedding(input_ids)

        if max_n_latents > 0:
            next_compute_range = (0, latent_positions[:, 1].min().item())

        # stop_logprob = []
        kv_cache = None
        latent_samples_all = [[] for _ in range(bs)]

        for pass_idx in range(max_n_latents if max_n_latents > 0 else 1):
            slice_start, slice_end = next_compute_range

            if kv_cache is None:
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, slice_start:slice_end, :],
                    attention_mask=attention_mask[:, slice_start:slice_end],
                    position_ids=(
                        position_ids[:, slice_start:slice_end]
                        if position_ids is not None else None
                    ),
                    output_hidden_states=True,
                )
                hidden_offset = 0
            else:
                past_key_values = [
                    (k[:, :, :slice_start, :], v[:, :, :slice_start, :])
                    for k, v in kv_cache
                ]
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, slice_start:slice_end, :],
                    attention_mask=attention_mask[:, :slice_end],
                    position_ids=(
                        position_ids[:, slice_start:slice_end]
                        if position_ids is not None else None
                    ),
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                )
                hidden_offset = slice_start

            kv_cache = outputs.past_key_values
            hidden_states = outputs.hidden_states[-1]

            next_compute_range = (
                slice_end,
                seq_len if pass_idx + 1 >= max_n_latents else slice_end + 1,
            )

            tensor_list = [
                [inputs_embeds[b, pos, :] for pos in range(seq_len)]
                for b in range(bs)
            ]

            for b, latent_positions in enumerate(latent_lists):
                if pass_idx >= len(latent_positions):
                    continue
                token_idx = latent_positions[pass_idx]
                vec = hidden_states[b, token_idx - 1 - hidden_offset, :]

                mu, logvar = self.latent_head.forward(vec)  # [dim*2]
                # print(mu, logvar)
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + std * eps
                # print(self.is_training)
                # dist = torch.distributions.Normal(mu, self.latent_sigma)
                # logprob = dist.log_prob(sample).sum()

                tensor_list[b][token_idx] = z
                latent_samples_all[b].append(z)

            inputs_embeds = torch.stack([torch.stack(tensor_list[b]) for b in range(bs)], dim=0)

            pred_token_id = self.end_head(inputs_embeds[:, next_compute_range[0], :])
            # print(torch.softmax(pred_token_id, dim=-1))
            if self.is_training == True:
                is_end = torch.multinomial(torch.softmax(pred_token_id, dim=-1), num_samples=1).squeeze(
                    -1).bool()  # [batch_size]
            else:
                is_end = (pred_token_id[:, 1] > pred_token_id[:, 0])
            # stop_logprob.append(torch.log_softmax(pred_token_id, dim=-1))
            if pass_idx % self.rl_config.c_thought == 0:
                attention_mask[is_end, bol_position + 1 + pass_idx:eol_position] = 0
            if pass_idx >= max_n_latents:  # or (attention_mask[:, bol_position + 1 + pass_idx] == 0).all():
                break

        self.gen_forward_cnt += max_n_latents + 1

        latent_dim = inputs_embeds.size(-1)
        latent_lengths = torch.tensor([len(lst) for lst in latent_lists],
                                      device=device, dtype=torch.long)
        max_latent_len = latent_lengths.max().item() if latent_lengths.numel() > 0 else 0

        if max_latent_len > 0:
            latent_inputs_embeds = torch.zeros(bs, max_latent_len, latent_dim, device=device)
            for b in range(bs):
                if latent_samples_all[b]:
                    latent_inputs_embeds[b, :len(latent_samples_all[b])] = torch.stack(latent_samples_all[b])
        else:
            latent_inputs_embeds = torch.zeros(bs, 0, latent_dim, device=device)

        return LatentForwardResult(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            latent_inputs_embeds=latent_inputs_embeds,
        )

    @torch.no_grad()
    def generate_with_latent(
            self,
            questions,
            questions_mask,
            max_new_tokens=16,
            temperature=1.0,
            synced_gpus=False,
    ):
        self.eval()
        device = questions.device
        batch_size = questions.size(0)

        latent_mask = (questions == self.latent_token_id)
        latent_positions = latent_mask.nonzero(as_tuple=False)
        latent_lists = [
            [idx[1].item() for idx in latent_positions if idx[0] == b]
            for b in range(batch_size)
        ]
        bol_position = latent_lists[0][0] - 1
        eol_position = latent_lists[0][-1] + 1
        questions = questions[:, :eol_position + 1]
        questions_mask = questions_mask[:, :eol_position + 1]
        position_ids = get_position_ids_from_attention_mask(questions_mask)

        latent_out = self.latent_forward_with_cache(
            input_ids=questions,
            attention_mask=questions_mask,
            position_ids=position_ids,
        )
        attention_mask = latent_out.attention_mask
        prompt_embeds = latent_out.inputs_embeds
        latent_inputs_embeds = latent_out.latent_inputs_embeds
        latent_attention_mask = attention_mask[:, bol_position + 1:eol_position]

        # 直接用 prompt_embeds 做一次全量前向，拿到最后 logits
        position_ids = get_position_ids_from_attention_mask(attention_mask)
        outputs = self.base_causallm(
            inputs_embeds=prompt_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        logits = outputs.logits[:, -1, :]

        tokens = [[questions[i, -1].item()] for i in range(batch_size)]
        unfinished = torch.ones(batch_size, dtype=torch.bool, device=device)
        cur_inputs_embeds = prompt_embeds
        cur_attention_mask = attention_mask

        for _ in range(max_new_tokens):
            if temperature != 1.0:
                logits = logits / temperature
            if self.is_training:
                probs = torch.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_tokens = torch.argmax(logits, dim=-1)

            next_tokens = torch.where(
                unfinished, next_tokens, torch.full_like(next_tokens, self.eos_token_id)
            )
            for i, tok in enumerate(next_tokens.tolist()):
                tokens[i].append(tok)

            next_token_embeds = self.embedding(next_tokens).unsqueeze(1)
            cur_inputs_embeds = torch.cat([cur_inputs_embeds, next_token_embeds], dim=1)
            cur_attention_mask = torch.cat(
                [cur_attention_mask, torch.ones(batch_size, 1, dtype=cur_attention_mask.dtype, device=device)],
                dim=1,
            )

            position_ids = get_position_ids_from_attention_mask(cur_attention_mask)
            outputs = self.base_causallm(
                inputs_embeds=cur_inputs_embeds,
                attention_mask=cur_attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
            self.gen_forward_cnt += 1
            logits = outputs.logits[:, -1, :]

            unfinished &= next_tokens.ne(self.eos_token_id)
            if not unfinished.any():
                break

        if synced_gpus:
            while self.gen_forward_cnt < max_new_tokens + MAX_N_LATENT:
                self.gen_forward_cnt += 1
                _ = self.base_causallm(
                    inputs_embeds=cur_inputs_embeds[:, cur_pos - 1:cur_pos, :],
                    attention_mask=cur_attention_mask[:, :cur_pos],
                    past_key_values=cur_kv_cache,
                    use_cache=True,
                )

        max_len = max(len(seq) for seq in tokens)
        pred_ids = torch.full(
            (batch_size, max_len),
            fill_value=self.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        for i, seq in enumerate(tokens):
            pred_ids[i, : len(seq)] = torch.tensor(seq, device=device)

        return GenerationOutputs(
            question_input_ids=questions[:, :bol_position + 1],
            question_attention_mask=questions_mask[:, :bol_position + 1],
            latent_inputs_embeds=latent_inputs_embeds,
            latent_attention_mask=latent_attention_mask,
            pred_ids=pred_ids,
        )

    def rl_step(self, batch, optimizer, writer=None, global_step=None):
        rl_config = self.rl_config
        device = self.device
        questions = batch["input_ids"]
        questions_mask = batch["attention_mask"]
        answers = batch["labels"]
        explainable_ids_list = batch["explainable_ids_list"]

        self.replay_buffer.clear()

        experience = self.rollout(
            questions=questions,
            questions_mask=questions_mask,
            group_size=rl_config.group_size,
            gt_answers=answers,
            explainable_ids_list=explainable_ids_list,
        )
        self.replay_buffer.append(experience.to("cpu"))

        log_dict = {
            "train/rewards": experience.rewards.mean().item(),
            "train/accuracies": experience.accuracies.mean().item(),
            "train/n_latent_forward": experience.n_latent_forward.float().mean().item(),
        }
        if getattr(experience, "explain_logprob", None) is not None:
            log_dict["train/explain_logprob"] = experience.explain_logprob.mean().item()

        torch.cuda.empty_cache()

        dataloader = DataLoader(
            dataset=self.replay_buffer,
            batch_size=rl_config.exp_batch_size,
            shuffle=True,
            collate_fn=grpo.join_experience_batch,
        )

        gradient_accumulation_steps = max(1, rl_config.gradient_accumulation_steps)
        total_inner_steps = len(dataloader)

        loop_sums = defaultdict(float)
        inner_steps = 0
        accum_counter = 0
        optimizer.zero_grad()
        last_grad_norm = 0.0

        for exp in dataloader:
            inner_steps += 1
            accum_counter += 1
            exp: grpo.Experience = exp.to(device)

            latent_logprobs, answer_logprobs = self.get_logprobs(e=exp)
            loss_dict = self.grpo_loss(
                latent_logprobs=latent_logprobs,
                answer_logprobs=answer_logprobs,
                experience=exp,
            )

            raw_total_loss = loss_dict["total_loss"]
            (raw_total_loss / gradient_accumulation_steps).backward()

            for k, v in loss_dict.items():
                value = v.item() if torch.is_tensor(v) else float(v)
                loop_sums[f"train/{k}"] += value

            need_update = (accum_counter % gradient_accumulation_steps == 0) or (inner_steps == total_inner_steps)
            if need_update:
                grad_norm = clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                last_grad_norm = grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)

        if inner_steps > 0:
            for key in list(loop_sums.keys()):
                log_dict[key] = loop_sums[key] / inner_steps
            log_dict["train/grad_norm"] = last_grad_norm

        return log_dict

    def extract_equal_result(self, expression):
        match = re.search(r'=(.*?)>>', expression)
        if match:
            return match.group(1).strip()
        else:
            return ''

    @torch.no_grad()
    def compute_explain_logprob(
            self,
            latent_inputs_embeds: torch.Tensor,  # (B, L_latent, H)
            n_latent_forward: torch.Tensor,  # (B,)
            pred_strings,
            max_gen_len=16
    ):
        if self.expainable_llm is None:
            return None

        device = self.expainable_llm.device
        eos_token_id = self.eos_token_id
        thought = self.rl_config.c_thought
        max_stage = self.rl_config.max_latent_stage
        tok = self.tokenizer  # 或 self.expainable_llm 的 tokenizer

        B, _, hidden = latent_inputs_embeds.size()

        # 1) 收集所有可用的 latent prefix
        prefix_list, owner = [], []
        for b in range(B):
            latent_valid = int(n_latent_forward[b].item())
            stage_num = min(max_stage, latent_valid // thought)
            for s in range(stage_num):
                start = s * thought
                end = start + thought
                prefix = latent_inputs_embeds[b, start:end, :].to(device)
                prefix_list.append(prefix)
                owner.append(b)

        N = len(prefix_list)
        if N == 0:
            return torch.zeros(B, device=device), torch.ones(B, device=device)

        seq_embeds = torch.stack(prefix_list, dim=0).clone()  # (N, thought, H)
        attention_mask = torch.ones(N, thought, dtype=torch.long, device=device)
        position_ids = torch.arange(thought, dtype=torch.long, device=device).unsqueeze(0).expand(N, -1)

        total_logprob = torch.zeros(N, device=device)
        total_tokens = torch.zeros(N, device=device)

        gen_token_ids = [[] for _ in range(N)]
        active_idx = torch.arange(N, device=device)
        # embed_layer = self.expainable_llm.get_input_embeddings()

        # 2) batched greedy decoding
        for _ in range(max_gen_len):
            outputs = self.expainable_llm(
                inputs_embeds=seq_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            logits = outputs.logits[:, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)

            next_token = torch.argmax(logits, dim=-1)
            step_logprob = log_probs.gather(-1, next_token.unsqueeze(-1)).squeeze(-1)

            total_logprob[active_idx] += step_logprob
            total_tokens[active_idx] += 1

            for aidx, tok_id in zip(active_idx.tolist(), next_token.tolist()):
                gen_token_ids[aidx].append(tok_id)

            eos_mask = next_token.eq(eos_token_id)
            keep_mask = ~eos_mask
            if keep_mask.any():
                kept_idx = active_idx[keep_mask]
                next_embed = self.embedding(next_token[keep_mask]).unsqueeze(1)
                seq_embeds = torch.cat([seq_embeds[keep_mask], next_embed], dim=1)

                attention_mask = torch.cat(
                    [attention_mask[keep_mask],
                     torch.ones(len(kept_idx), 1, dtype=torch.long, device=device)],
                    dim=1
                )
                next_pos = position_ids[keep_mask][:, -1:] + 1
                position_ids = torch.cat([position_ids[keep_mask], next_pos], dim=1)

                active_idx = kept_idx
            else:
                break

        # 3) 汇总到 batch 维度
        explain_logprob_b = torch.zeros(B, device=device)
        explain_coherence_b = torch.ones(B, device=device)
        explain_count_b = torch.zeros(B, dtype=torch.long, device=device)
        owner_t = torch.tensor(owner, dtype=torch.long, device=device)

        valid_mask = total_tokens > 0
        avg_logprob = torch.zeros_like(total_logprob)
        avg_logprob[valid_mask] = total_logprob[valid_mask]

        for idx in range(N):
            b = owner_t[idx].item()
            explain_logprob_b[b] += avg_logprob[idx]
            explain_count_b[b] += 1

        explain_logprob_b = torch.where(
            explain_count_b > 0,
            explain_logprob_b / explain_count_b.clamp_min(1),
            torch.zeros_like(explain_logprob_b)
        )

        # 4) 解码并打印
        preview_by_batch = [[] for _ in range(B)]
        result_by_batch = [[] for _ in range(B)]
        for idx, tlist in enumerate(gen_token_ids):
            txt = tok.decode(tlist, skip_special_tokens=True) if tlist else ""
            b = owner_t[idx].item()
            result_of_eq = self.extract_equal_result(txt)
            preview_by_batch[b].append(txt)
            result_by_batch[b].append(result_of_eq)

        for b in range(B):
            total_steps = len(result_by_batch[b])  # eg. ["4", "3", "3"]
            if total_steps == 0:
                explain_coherence_b[b] = 1.0
                continue

            matched = 0
            for i in range(len(result_by_batch[b]) - 1):
                for j in range(i + 1, len(preview_by_batch[b])):
                    if result_by_batch[b][i] in preview_by_batch[b][j]:
                        matched += 1
                        break
            last_result = result_by_batch[b][-1]
            if last_result in pred_strings[b]:
                matched += 1
            explain_coherence_b[b] = matched / total_steps
        # for b, seqs in enumerate(preview_by_batch):
        #     print(f"[sample {b}]", end=' ')
        #     for j, s in enumerate(seqs):
        #         print(f"step {j}: {s}", end=' ')
        #     print()
        #
        # for b, seqs in enumerate(result_by_batch):
        #     print(f"[sample {b}]", end=' ')
        #     for j, s in enumerate(seqs):
        #         print(f"step {j}: {s}", end=' ')
        #     print()
        # 只返回 logprob
        return explain_logprob_b, explain_coherence_b

    @torch.no_grad()
    def rollout(self, questions, questions_mask, gt_answers, group_size, explainable_ids_list=None):
        rl_cfg = self.rl_config
        group_size = group_size

        # 扩展 group
        group_questions = questions.repeat_interleave(group_size, dim=0)
        group_masks = questions_mask.repeat_interleave(group_size, dim=0)
        if explainable_ids_list is not None:
            group_explain_ids = [copy.deepcopy(ids) for ids in explainable_ids_list for _ in range(group_size)]
        else:
            group_explain_ids = None

        gen_out = self.generate_with_latent(
            questions=group_questions.to(self.device),
            questions_mask=group_masks.to(self.device),
            max_new_tokens=rl_cfg.max_new_tokens,
            temperature=rl_cfg.temperature,
        )
        # print(gen_out.pred_ids.tolist())
        pred_strings = self.tokenizer.batch_decode(gen_out.pred_ids, skip_special_tokens=True)
        n_latent_forward = gen_out.latent_attention_mask.sum(dim=1)

        explain_logprob, explain_coherence = None, None
        if self.expainable_llm is not None and group_explain_ids and rl_cfg.reason_reward_weight != 0:
            explain_logprob, explain_coherence = self.compute_explain_logprob(
                latent_inputs_embeds=gen_out.latent_inputs_embeds,
                n_latent_forward=n_latent_forward,
                pred_strings=pred_strings,
            )
            # print(explain_logprob.size())

        rewards_list, acc_list, adv_list = [], [], []
        for i in range(len(questions)):
            start = i * group_size
            end = (i + 1) * group_size
            rewards, accuracies = self.get_group_rewards_and_acc(
                pred_answers=pred_strings[start:end],
                gt_answer=gt_answers[i],
                explain_coherence=None if explain_coherence is None else explain_coherence[start:end],
                n_latent_forward=n_latent_forward[start:end],
                explain_logprob_chunk=None if explain_logprob is None else explain_logprob[start:end],
            )
            rewards_list.append(rewards)
            acc_list.append(accuracies)
            adv_list.append(grpo.group_advantages(rewards))

        rewards = torch.cat(rewards_list, dim=0)
        accuracies = torch.cat(acc_list, dim=0)
        advantages = torch.cat(adv_list, dim=0)
        # print(gen_out.latent_inputs_embeds.size())
        # print(gen_out.latent_attention_mask.size())
        # print(gen_out.question_input_ids.size())
        # print(gen_out.question_attention_mask.size())
        experience = grpo.Experience(
            question_input_ids=gen_out.question_input_ids,
            question_attention_mask=gen_out.question_attention_mask,
            latent_inputs_embeds=gen_out.latent_inputs_embeds,
            latent_attention_mask=gen_out.latent_attention_mask,
            answer_input_ids=gen_out.pred_ids,
            answer_attention_mask=gen_out.pred_ids.ne(self.pad_token_id).long(),
            n_latent_forward=n_latent_forward.unsqueeze(1),
            rewards=rewards,
            accuracies=accuracies,
            advantages=advantages,
        )

        if explain_logprob is not None:
            experience.explain_logprob = explain_logprob

        latent_logprobs, answer_logprobs = self.get_logprobs(experience)
        # print(latent_logprobs.size())
        # print(advantages.size())
        experience.latent_logprobs = latent_logprobs
        experience.answer_logprobs = answer_logprobs
        return experience

    def get_logprobs(self, e: grpo.Experience):
        question_length = e.question_input_ids.shape[1]
        latent_length = e.latent_inputs_embeds.shape[1]
        answer_length = e.answer_input_ids.shape[1]

        question_inputs_embeds = self.embedding(e.question_input_ids)
        answer_inputs_embeds = self.embedding(e.answer_input_ids)

        all_inputs_embeds = torch.cat(
            [question_inputs_embeds, e.latent_inputs_embeds, answer_inputs_embeds], dim=1
        )
        all_attention_mask = torch.cat(
            [e.question_attention_mask, e.latent_attention_mask, e.answer_attention_mask], dim=1
        )
        all_position_ids = get_position_ids_from_attention_mask(all_attention_mask)

        all_outputs = self.base_causallm.forward(
            inputs_embeds=all_inputs_embeds,
            attention_mask=all_attention_mask,
            position_ids=all_position_ids,
            output_hidden_states=True,
        )

        # ===== Latent log_prob Gaussian =====
        hidden = all_outputs.hidden_states[-1]
        vec = hidden[:, question_length - 1: question_length + latent_length - 1, :]

        mu, logvar = self.latent_head.forward(vec)  # [dim*2]
        std = torch.exp(0.5 * logvar)
        dist = torch.distributions.Normal(mu, std)
        latent_samples = e.latent_inputs_embeds[:, :latent_length, :]
        pred_token_id = self.end_head(latent_samples)
        # print(torch.softmax(pred_token_id, dim=-1))
        for pass_idx in range(e.latent_attention_mask.size(1)):
            if pass_idx % self.rl_config.c_thought:
                pred_token_id[:, pass_idx, 1] = float('-inf')
        pred_logprob = torch.log_softmax(pred_token_id, dim=-1)
        eos_place = e.latent_attention_mask.sum(-1)[:, None]
        # print(eos_place)
        log_prob_fill = torch.cat((pred_logprob[:, :, 1], torch.zeros_like(pred_logprob[:, 0:1, 1])), dim=1)
        latent_logprobs = dist.log_prob(latent_samples).sum(dim=-1) + pred_logprob[:, :, 0]
        # print(latent_logprobs.tolist())
        logits = all_outputs.logits
        answer_logits = logits[:, question_length + latent_length - 1: -1, :]
        answer_logprobs = F.log_softmax(answer_logits, dim=-1)
        answer_logprobs = answer_logprobs.gather(dim=-1, index=e.answer_input_ids.unsqueeze(-1)).squeeze(-1)
        answer_logprobs[:, 0] = log_prob_fill.gather(dim=1, index=eos_place).squeeze(-1)
        # print(answer_logprobs.tolist())

        return latent_logprobs, answer_logprobs

    def get_group_rewards_and_acc(
            self,
            pred_answers,
            gt_answer,
            explain_coherence,
            n_latent_forward,
            explain_logprob_chunk=None,
    ):
        rl_config = self.rl_config
        group_size = len(pred_answers)
        format = torch.zeros(group_size, 1, device=self.device)
        accuracies = torch.zeros(group_size, 1, device=self.device)
        for i, ans in enumerate(pred_answers):
            gt_answer[gt_answer == -100] = self.pad_token_id
            gt_string = self.tokenizer.decode(gt_answer, skip_special_tokens=True)
            marker = "<|end-latent|>###"
            gt_answer_string = gt_string[gt_string.rfind("###") + len("###"):]
            idx = ans.rfind(marker)
            if idx == -1:
                pred_string = ""
            else:
                pred_string = ans[idx + len(marker):]
                if len(pred_string.strip()) != 0:
                    format[i] = 1
            accuracies[i] = self.verify_answer(gt_answer=gt_answer_string, pred_answer=pred_string)
            print(
                f"Ground_truth: {gt_answer_string} Answer {ans} Accuracy: {accuracies[i]} Format: {format[i]}")
            if explain_logprob_chunk is not None:
                print(f"Explain_prob: {explain_logprob_chunk[i]}")
            if explain_coherence is not None:
                print(f"Explain_coherence: {explain_coherence[i]}")

        rewards = rl_config.accuracy_reward_weight * accuracies.detach().clone() + rl_config.format_reward_weight * format.detach().clone()

        # if explain_logprob_chunk is not None:
        #     rewards = rewards + rl_config.reason_reward_weight * explain_logprob_chunk[:, None]
        if explain_coherence is not None:
            if rl_config.accuracy_reward_weight ==0:
                rewards = rewards + rl_config.reason_reward_weight * explain_coherence[:, None]
            else:
                rewards = rewards + rl_config.reason_reward_weight * explain_coherence[:, None] * (accuracies.detach().clone())

        return rewards, accuracies

    def verify_answer(self, gt_answer: str, pred_answer: str) -> float:
        # gt_answer = gt_answer.strip("\n ").rstrip(".").replace(",", "")
        # pred_answer = pred_answer.strip("\n ").rstrip(".").rstrip(".\n").replace(",", "")
        try:  # some answers may be like '10.0' but predicted as '10'
            gt_answer = float(gt_answer)
            pred_answer = float(pred_answer)
            return float(gt_answer == pred_answer)
        except ValueError:
            return float(self.is_equiv(gt_answer, pred_answer))

    @torch.no_grad()
    def eval_generation(self, dataloader, val='val'):
        loop_sums = defaultdict(float)
        inner_steps = 0
        self.is_training = False
        for step, batch in enumerate(dataloader):
            inner_steps += 1
            questions = batch["input_ids"]
            questions_mask = batch["attention_mask"]
            answers = batch["labels"]
            explainable_ids_list = batch["explainable_ids_list"]

            experience = self.rollout(
                questions=questions,
                questions_mask=questions_mask,
                gt_answers=answers,
                group_size=1,
                explainable_ids_list=explainable_ids_list,
            )

            loop_sums[f"{val}/rewards"] += experience.rewards.mean().item()
            loop_sums[f"{val}/accuracies"] += experience.accuracies.mean().item()
            loop_sums[f"{val}/n_latent_forward"] += experience.n_latent_forward.float().mean().item()
            if getattr(experience, "explain_logprob", None) is not None:
                loop_sums[f"{val}/explain_logprob"] += experience.explain_logprob.mean().item()

            torch.cuda.empty_cache()
        print(f"Accuracy in Validation {loop_sums[f'{val}/accuracies'] * 100 / inner_steps}")
        log_dict = {}
        if inner_steps > 0:
            for key in loop_sums:
                log_dict[key] = loop_sums[key] / inner_steps

        self.is_training = True
        return log_dict

        # -- evaluation ends --#

    def is_equiv(self, str1, str2, verbose=False):
        if str1 is None and str2 is None:
            print("WARNING: Both None")
            return True
        if str1 is None or str2 is None:
            return False

        try:
            ss1 = self.strip_string(str1)
            ss2 = self.strip_string(str2)
            if verbose:
                print(ss1, ss2)
            return ss1 == ss2
        except Exception:
            return str1 == str2

    def remove_boxed(self, s):
        if "\\boxed " in s:
            left = "\\boxed "
            assert s[:len(left)] == left
            return s[len(left):]

        left = "\\boxed{"

        assert s[:len(left)] == left
        assert s[-1] == "}"

        return s[len(left):-1]

    def last_boxed_only_string(self, string):
        idx = string.rfind("\\boxed")
        if "\\boxed " in string:
            return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
        if idx < 0:
            idx = string.rfind("\\fbox")
            if idx < 0:
                return None

        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(string):
            if string[i] == "{":
                num_left_braces_open += 1
            if string[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1

        if right_brace_idx is None:
            retval = None
        else:
            retval = string[idx:right_brace_idx + 1]

        return retval

    def fix_fracs(self, string):
        substrs = string.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += "\\frac"
                if substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except AssertionError:
                        return string
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}" + b + post_substr
                        else:
                            new_str += "{" + a + "}" + b
        string = new_str
        return string

    def fix_a_slash_b(self, string):
        if len(string.split("/")) != 2:
            return string
        a = string.split("/")[0]
        b = string.split("/")[1]
        try:
            a = int(a)
            b = int(b)
            assert string == "{}/{}".format(a, b)
            new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
            return new_string
        except AssertionError:
            return string

    def remove_right_units(self, string):
        # "\\text{ " only ever occurs (at least in the val set) when describing units
        if "\\text{ " in string:
            splits = string.split("\\text{ ")
            assert len(splits) == 2
            return splits[0]
        else:
            return string

    def fix_sqrt(self, string):
        if "\\sqrt" not in string:
            return string
        splits = string.split("\\sqrt")
        new_string = splits[0]
        for split in splits[1:]:
            if split[0] != "{":
                a = split[0]
                new_substr = "\\sqrt{" + a + "}" + split[1:]
            else:
                new_substr = "\\sqrt" + split
            new_string += new_substr
        return new_string

    def strip_string(self, string):
        # linebreaks
        string = string.replace("\n", "")

        # remove inverse spaces
        string = string.replace("\\!", "")

        # replace \\ with \
        string = string.replace("\\\\", "\\")

        # replace tfrac and dfrac with frac
        string = string.replace("tfrac", "frac")
        string = string.replace("dfrac", "frac")

        # remove \left and \right
        string = string.replace("\\left", "")
        string = string.replace("\\right", "")

        # Remove circ (degrees)
        string = string.replace("^{\\circ}", "")
        string = string.replace("^\\circ", "")

        # remove dollar signs
        string = string.replace("\\$", "")

        # remove units (on the right)
        string = self.remove_right_units(string)

        # remove percentage
        string = string.replace("\\%", "")
        string = string.replace("\%", "")  # noqa: W605

        # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
        string = string.replace(" .", " 0.")
        string = string.replace("{.", "{0.")
        # if empty, return empty string
        if len(string) == 0:
            return string
        if string[0] == ".":
            string = "0" + string

        # to consider: get rid of e.g. "k = " or "q = " at beginning
        if len(string.split("=")) == 2:
            if len(string.split("=")[0]) <= 2:
                string = string.split("=")[1]

        # fix sqrt3 --> sqrt{3}
        string = self.fix_sqrt(string)

        # remove spaces
        string = string.replace(" ", "")

        # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
        string = self.fix_fracs(string)

        # manually change 0.5 --> \frac{1}{2}
        if string == "0.5":
            string = "\\frac{1}{2}"

        # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
        string = self.fix_a_slash_b(string)

        return string
