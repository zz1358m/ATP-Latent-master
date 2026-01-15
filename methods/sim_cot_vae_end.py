# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import math
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
from modules.projector import STOPPolicy, LatentPolicy
from modules.utils import get_position_ids_from_attention_mask
import copy

Outputs = namedtuple("Outputs", ["loss", "loss_explain_all", "inputs_embeds", "logits"])
Outputs_withkl = namedtuple("Outputs", ["loss", "loss_explain_all", "loss_kl", "inputs_embeds", "logits"])
Outputs_withmask = namedtuple("Outputs_withmask",
                              ["loss", "attention_mask", "loss_explain_all", "loss_kl", "loss_stop", "inputs_embeds",
                               "logits"])
MAX_N_LATENT = 8


class CoconutGPT_Same_Word_Embedding_EndSignal_VAE(nn.Module):
    def __init__(
            self,
            base_causallm,
            expainable_llm,
            # for debug
            tokenizer,
            latent_token_id,
            start_latent_id,
            end_latent_id,
            eos_token_id,
            step_start_id,
            c_thought,
            configs,
    ):

        super(CoconutGPT_Same_Word_Embedding_EndSignal_VAE, self).__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.base_causallm.config.use_cache = True
        self.end_head = STOPPolicy(
            feature_size=self.base_causallm.config.hidden_size,
            intermediate_size=self.base_causallm.config.hidden_size
        )
        self.latent_head = LatentPolicy(
            feature_size=self.base_causallm.config.hidden_size,
            intermediate_size=self.base_causallm.config.hidden_size,
            deterministic=False
        )
        self.expainable_llm = expainable_llm
        # self.expainable_llm.config.use_cache = True
        self.tokenizer = tokenizer
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.step_start_id = step_start_id
        self.c_thought = c_thought
        self.config = configs

        if hasattr(self.config, "training_method"):
            if self.config.training_method == 'only_expainable_llm':
                for param in self.base_causallm.parameters():
                    param.requires_grad = False
            elif self.config.training_method == 'only_base_causallm':
                for param in self.expainable_llm.parameters():
                    param.requires_grad = False
            elif self.config.training_method == 'full':
                pass
            elif self.config.training_method == 'freeze_backbone':
                for param in self.base_causallm.parameters():
                    param.requires_grad = False

                for param in self.expainable_llm.parameters():
                    param.requires_grad = False
            else:
                raise ValueError(f"not this training_method {self.config.training_method=}")

        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
            print("is GPT")
        else:
            self.embedding = self.base_causallm.get_input_embeddings()
            print("is not GPT")

    def forward(self, input_ids, attention_mask, labels, position_ids, decoding=False, **kwargs):

        logits = []
        loss = 0.0
        loss_stop = 0.0
        loss_explain_all = torch.tensor(0.0, device=input_ids.device)
        kl_loss = torch.tensor(0.0, device=input_ids.device)
        c_thought_num = 1
        latent_indices = (
                input_ids == self.latent_token_id
        ).nonzero()  # (num_latent_tokens_in_the_batch, 2)

        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]  # bs, num_latent_tokens_in_the_instance (difference across the batch)

        bol_position = latent_lists[0][0] - 1
        eol_position = latent_lists[0][-1] + 1
        max_n_latents = max([len(l) for l in latent_lists])

        latent_mu_collector, latent_logvar_collector, latent_n_collector = [], [], []
        next_compute_range = (0, input_ids.shape[1])
        inputs_embeds = self.embedding(input_ids)

        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())
            # before the earliest latent token position
        if hasattr(self.config, 'explain_mode') and self.config.explain_mode == 'v1_aug':
            # print('explainable_ids_list' in kwargs)
            if 'explainable_ids_list' in kwargs or decoding:
                c_thought_num = len(latent_lists[0]) // self.c_thought

                input_united_tokens = []

                def safe_token_id(x):
                    return x[0] if isinstance(x, list) else x

                start_token = safe_token_id(self.tokenizer.encode('<<', add_special_tokens=False))  # "<<"
                end_token = safe_token_id(self.tokenizer.encode('>>', add_special_tokens=False))  # ">>"
                separator_token = safe_token_id(self.tokenizer.encode('\n', add_special_tokens=False))  # "\n"

                def trim_trailing_zeros(group):
                    while group and group[-1] == 0:
                        group.pop()
                    return group

                def replace_llama_special_tokens(x, merged_token, end_token, separator_token):
                    out = []
                    for seq in x:
                        new_seq = []
                        for t in seq:
                            if t.item() == merged_token:
                                new_seq.extend([end_token, separator_token])
                            elif t.item() != 0 or len(new_seq) > 0:
                                new_seq.append(t.item())
                        out.append(torch.tensor(new_seq, device=x.device))
                    return out

                if len(self.tokenizer.encode('>>\n', add_special_tokens=False)) == 1 and not decoding:
                    merge_token = self.tokenizer.encode('>>\n', add_special_tokens=False)[0]
                    kwargs['explainable_ids_list'] = copy.deepcopy(
                        replace_llama_special_tokens(kwargs['explainable_ids_list'], merge_token, end_token,
                                                     separator_token))
                if not decoding:
                    for j, seq in enumerate(kwargs['explainable_ids_list']):
                        i = 0
                        groups = []
                        while i < len(seq):
                            if seq[i] == start_token:

                                group = [start_token]
                                i += 1
                                while i < len(seq):
                                    group.append(seq[i])
                                    if seq[i] == end_token:
                                        break
                                    i += 1
                                group = trim_trailing_zeros(group)
                                groups.append(group)
                            else:
                                i += 1
                        print(len(groups))

                        input_united_groups = []
                        combined_group = []
                        group_count = 0

                        for group in groups:
                            group_count += 1
                            if group_count <= self.config.max_latent_stage - 1:
                                group = [-570] * self.c_thought + group + [self.eos_token_id]
                                cleaned_group = [int(x) if torch.is_tensor(x) else x for x in group]
                                input_united_groups.append(cleaned_group)
                            else:
                                if combined_group and combined_group[-1] == end_token and group[0] == start_token:
                                    combined_group.append(separator_token)
                                combined_group.extend(group)

                        if len(groups) < c_thought_num:
                            attention_mask[j, bol_position + len(groups) * self.config.c_thought + 1: eol_position] = 0
                            padding_group = [-570] * self.c_thought + [self.eos_token_id]
                            input_united_groups.extend(
                                [padding_group.copy() for _ in range(c_thought_num - len(groups))])

                        if combined_group:
                            final_group = [-570] * self.c_thought + combined_group + [self.eos_token_id]
                            cleaned_group = [int(x) if torch.is_tensor(x) else x for x in final_group]
                            input_united_groups.append(cleaned_group)

                        input_united_tokens.append(copy.deepcopy(input_united_groups))

                ## max pad len

                kv_cache = None
                position_ids = get_position_ids_from_attention_mask(attention_mask)
                for pass_idx in range(max_n_latents):

                    if kv_cache == None:
                        # first forward pass
                        outputs = self.base_causallm(
                            inputs_embeds=inputs_embeds[
                                          :, next_compute_range[0]: next_compute_range[1], :
                                          ],
                            attention_mask=attention_mask[
                                           :, next_compute_range[0]: next_compute_range[1]
                                           ],
                            position_ids=position_ids[
                                         :, next_compute_range[0]: next_compute_range[1]
                                         ],
                            output_hidden_states=True,
                        )
                        hidden_states_offset = 0

                    else:
                        # extract kv cache to reuse
                        past_key_values = [
                            (
                                k[:, :, : next_compute_range[0], :],
                                v[:, :, : next_compute_range[0], :],
                            )
                            for k, v in kv_cache
                        ]

                        outputs = self.base_causallm(
                            inputs_embeds=inputs_embeds[
                                          :, next_compute_range[0]: next_compute_range[1], :
                                          ],
                            attention_mask=attention_mask[:, : next_compute_range[1]],
                            position_ids=position_ids[
                                         :, next_compute_range[0]: next_compute_range[1]
                                         ],
                            past_key_values=past_key_values,
                            output_hidden_states=True,
                        )

                        hidden_states_offset = next_compute_range[0]
                        # when we use kv_cache for the first k tokens
                        # in `outputs.hidden_states`, [0, k) will be skipped
                        # so we need to keep this offset to correctly use the last hidden states

                    logits.append(outputs.logits)

                    next_compute_range = (
                        next_compute_range[1],
                        (
                            input_ids.shape[1]
                            if pass_idx + 1 >= max_n_latents
                            else next_compute_range[1] + 1
                        ),
                    )
                    hidden_states = outputs.hidden_states[
                        -1
                    ]  # Get the last layer hidden states
                    kv_cache = outputs.past_key_values
                    # pred_token_id = self.end_head(hidden_states[:, -1])
                    # print(hidden_states.size(), pred_token_id.size())
                    # if decoding == True:
                    #     is_end = (pred_token_id[:, 1] > pred_token_id[:, 0])  # [batch_size]
                    #     print(pass_idx, torch.softmax(pred_token_id, dim=-1))
                    #     if pass_idx % self.config.c_thought == 0:
                    #         attention_mask[is_end, next_compute_range[1]:eol_position] = 0
                    # else:
                    #     label_stop = (attention_mask[:, next_compute_range[0]] == 0).long()
                    #     print(label_stop)
                    #     loss_stop += F.cross_entropy(pred_token_id, label_stop)
                    # feedback the continuous thoughts to the input_embeds

                    # first decide the positions to feedback
                    filling_indices = [
                        (instance_idx, mask_list[pass_idx])
                        for instance_idx, mask_list in enumerate(latent_lists)
                        if len(mask_list) > pass_idx
                    ]

                    # to avoid in-place operations
                    # break down inputs_embeds (bs, len, hidden_size) into a list of list of 1-d tensors
                    tensor_list = [
                        [
                            inputs_embeds[batch_idx, pos, :]
                            for pos in range(inputs_embeds.shape[1])
                        ]
                        for batch_idx in range(inputs_embeds.shape[0])
                    ]

                    # replace some of them with continuous thoughts
                    for idx_pair in filling_indices:
                        batch_idx, token_idx = idx_pair

                        vec = hidden_states[batch_idx, token_idx - 1 - hidden_states_offset, :]  # [dim]

                        mu, logvar = self.latent_head.forward(vec)  # [dim*2]
                        std = torch.exp(0.5 * logvar)
                        eps = torch.randn_like(std)
                        z = mu + std * eps
                        tensor_list[batch_idx][token_idx] = z

                        latent_mu_collector.append(mu)
                        latent_logvar_collector.append(logvar)
                    # assemble the new inputs_embeds
                    inputs_embeds = torch.stack(
                        [
                            torch.stack(tensor_list[batch_idx])
                            for batch_idx in range(inputs_embeds.shape[0])
                        ]
                    )

                    pred_token_id = self.end_head(inputs_embeds[:, next_compute_range[0], :])
                    print(inputs_embeds.size(), pred_token_id.size())
                    if decoding == True:
                        is_end = (pred_token_id[:, 1] > pred_token_id[:, 0])  # [batch_size]
                        print(pass_idx, torch.softmax(pred_token_id, dim=-1))
                        if pass_idx % self.config.c_thought == 0:
                            attention_mask[is_end, bol_position + 1 + pass_idx:eol_position] = 0
                    else:
                        label_stop = (attention_mask[:, next_compute_range[0]] == 0).long()
                        print(label_stop)
                        loss_stop += F.cross_entropy(pred_token_id, label_stop)
                # final pass
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                                  :, next_compute_range[0]: next_compute_range[1], :
                                  ],
                    attention_mask=attention_mask[:, : next_compute_range[1]],
                    position_ids=position_ids[:, next_compute_range[0]: next_compute_range[1]],
                    past_key_values=(
                        [
                            (
                                k[:, :, : next_compute_range[0], :],
                                v[:, :, : next_compute_range[0], :],
                            )
                            for k, v in kv_cache
                        ]
                        if kv_cache
                        else None
                    ),
                    output_hidden_states=True,
                )

                logits.append(outputs.logits)

                self.gen_forward_cnt += max_n_latents + 1

                logits = torch.cat(logits, dim=-2)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = CrossEntropyLoss()
                if self.config.training_method == 'only_base_causallm' or self.config.training_method == 'full':
                    loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                    )
                    prior_std = 1
                    prior_var = prior_std ** 2
                    mus = torch.stack(latent_mu_collector)  # [num_latent, dim]
                    logvars = torch.stack(latent_logvar_collector)

                    kl = 0.5 * ((mus ** 2 + logvars.exp()) / prior_var - 1 - logvars + math.log(prior_var))

                    kl_loss = kl.mean()
                    if hasattr(self.config, 'kl_factor'):
                        loss = loss + self.config.kl_factor * kl_loss
                    else:
                        loss = loss + 0.001 * kl_loss

                if hasattr(self.config, 'visualize') and self.config.visualize:
                    debug_predictions = []

                    for debug_idx in range(0, len(latent_lists[0]), self.config.c_thought):

                        continuous_embeds = inputs_embeds[:, latent_lists[0][debug_idx: debug_idx + self.c_thought],
                                            :].to(
                            self.expainable_llm.device)

                        if hasattr(self.config, 'w_prompt') and self.config.w_prompt:
                            if hasattr(self.config, 'explain_mode') and self.config.explain_mode == 'v1_aug':
                                thought_idx = debug_idx // 2
                                if thought_idx != 2:
                                    input_explain_input_embeds_pre_order_prompt_ids = self.tokenizer(
                                        f'Step {thought_idx + 1} of the solution', add_special_tokens=False).input_ids
                                else:
                                    input_explain_input_embeds_pre_order_prompt_ids = self.tokenizer(
                                        f'Step 3 and all the remaining steps of the solution',
                                        add_special_tokens=False).input_ids
                                bz = continuous_embeds.shape[0]
                                input_explain_input_embeds_pre_order_prompt_embeds = self.embedding(
                                    torch.tensor(input_explain_input_embeds_pre_order_prompt_ids).to(
                                        self.expainable_llm.device))[None, ...].repeat(bz, 1, 1)
                                continuous_embeds = torch.cat(
                                    [input_explain_input_embeds_pre_order_prompt_embeds, continuous_embeds], dim=1)
                        debug_ids = torch.empty((1, 0), dtype=torch.long, device=self.expainable_llm.device)
                        while True:
                            if debug_ids.shape[0] != 0:
                                debug_embeds = torch.cat([continuous_embeds, self.embedding(debug_ids)], dim=1)
                            else:
                                debug_embeds = continuous_embeds
                            explainable_outputs = self.expainable_llm(
                                inputs_embeds=debug_embeds,
                                attention_mask=torch.ones(debug_embeds.shape[:2]).to(self.expainable_llm.device),
                                position_ids=torch.arange(1, debug_embeds.shape[1] + 1).unsqueeze(dim=0).to(
                                    self.expainable_llm.device),
                                output_hidden_states=True,
                            )
                            debug_logits = explainable_outputs.logits[:, -1, :] / .98
                            probs = torch.softmax(debug_logits, dim=-1)
                            next_token = torch.multinomial(probs, num_samples=1)
                            debug_ids = torch.cat([debug_ids, next_token], dim=1)

                            if torch.all(next_token == self.eos_token_id) or debug_ids.shape[
                                -1] > 512:  # 为 <eos>或者是'>>'结束
                                break

                        print(self.tokenizer.decode(debug_ids[0]))
                        debug_predictions.append(self.tokenizer.decode(debug_ids[0]))

                    if hasattr(self.config, 'visualize_jsonl') and self.config.visualize_jsonl != '':
                        save_jsonl_line(self.config.visualize_jsonl, {"predictiion": debug_predictions})

                if not decoding:
                    bz = len(input_united_tokens)

                    for thought_idx in range(c_thought_num):

                        max_pad_len = max(len(input_united_tokens[bz_idx][thought_idx]) for bz_idx in range(bz))
                        max_pad_len += 1
                        for bz_idx in range(bz):
                            token_seq = input_united_tokens[bz_idx][thought_idx]
                            pad_len = max_pad_len - len(token_seq)
                            if pad_len > 0:
                                token_seq += [self.eos_token_id] * pad_len
                                input_united_tokens[bz_idx][thought_idx] = token_seq

                    print("there")
                    print("there2")
                    for thought_idx in range(c_thought_num):
                        input_explain_input_embeds = []
                        input_explain_attention_mask, input_explain_position_ids, input_explain_labels = [], [], []
                        max_pad_len = -1

                        for bz_idx in range(bz):
                            latent_len = len(latent_lists[bz_idx])
                            start_idx = thought_idx * self.c_thought
                            end_idx = min(start_idx + self.c_thought, latent_len)
                            continuous_embeds = inputs_embeds[bz_idx, latent_lists[bz_idx][start_idx:end_idx], :]

                            other_embeds = self.embedding(
                                torch.tensor(input_united_tokens[bz_idx][thought_idx][self.c_thought:]).to(
                                    self.expainable_llm.device))
                            input_explain_input_embeds.append(torch.cat([continuous_embeds, other_embeds], dim=0))
                            attention_eos_index = input_united_tokens[bz_idx][thought_idx].index(self.eos_token_id)
                            attention_explain_mask = torch.zeros(len(input_united_tokens[bz_idx][thought_idx]),
                                                                 dtype=int)
                            attention_explain_mask[:attention_eos_index + 1] = 1
                            input_explain_attention_mask.append(attention_explain_mask)
                            input_explain_position_ids.append(
                                torch.arange(1, len(input_united_tokens[bz_idx][thought_idx]) + 1, dtype=int))
                            explain_labels = torch.tensor(input_united_tokens[bz_idx][thought_idx], dtype=int)
                            explain_labels_mask = (explain_labels != -570) & (explain_labels != self.eos_token_id)
                            explain_labels_mask[attention_eos_index] = True
                            explain_labels[~explain_labels_mask] = -100
                            input_explain_labels.append(explain_labels)

                        input_explain_input_embeds = torch.stack(input_explain_input_embeds)
                        input_explain_attention_mask = torch.stack(input_explain_attention_mask)
                        input_explain_position_ids = torch.stack(input_explain_position_ids)
                        input_explain_labels = torch.stack(input_explain_labels)
                        # print(input_explain_input_embeds.size())

                        explainable_outputs = self.expainable_llm(
                            inputs_embeds=input_explain_input_embeds.to(self.expainable_llm.device),
                            attention_mask=input_explain_attention_mask.to(self.expainable_llm.device),
                            position_ids=input_explain_position_ids.to(self.expainable_llm.device),
                            output_hidden_states=True,
                        )
                        if hasattr(self.config, "use_prj") and self.config.use_prj:
                            explainable_logits = self.base_causallm.lm_head(
                                self.projector2(explainable_outputs.hidden_states[-1]))
                        else:
                            explainable_logits = explainable_outputs.logits

                        if hasattr(self.config, "loss_level") and self.config.loss_level == 'token_level':
                            effective_token_count = (input_explain_labels != -100).sum()
                        else:
                            effective_token_count = float((input_explain_labels != -100).sum(dim=1).bool().sum().item())

                        shift_explain_logits = explainable_logits[..., :-1, :].contiguous()
                        shift_explain_labels = input_explain_labels[..., 1:].contiguous()
                        loss_explain_fct = CrossEntropyLoss(reduction='sum')
                        loss_explain = loss_explain_fct(
                            shift_explain_logits.view(-1, shift_explain_logits.size(-1)).to(self.expainable_llm.device),
                            shift_explain_labels.view(-1).to(self.expainable_llm.device)
                        )
                        loss_explain /= effective_token_count
                        loss_explain_all += loss_explain

        if 'explainable_ids_list' in kwargs:
            if loss is None:
                loss = 0.0
            # print(loss_explain_all)
            loss += 1.0 * loss_explain_all / c_thought_num
            loss += 1.0 * loss_stop / c_thought_num

        return Outputs_withmask(loss=loss, attention_mask=attention_mask,
                                loss_explain_all=loss_explain_all / c_thought_num,
                                loss_kl=kl_loss / c_thought_num,
                                loss_stop=loss_stop / c_thought_num,
                                inputs_embeds=inputs_embeds, logits=logits)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_causallm.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def generate(
            self,
            input_ids,
            attention_mask,  # attention_mask is not used
            max_new_tokens=16,
            output_embedding=False,
            synced_gpus=False,
            **kwargs
    ):

        self.gen_forward_cnt = 0

        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"

        tokens = input_ids[0].detach().tolist()

        labels = input_ids.clone()  # placeholder. not used.
        outputs = self.forward(
            input_ids,
            torch.ones_like(input_ids, device=input_ids.device),
            labels,
            torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).reshape(1, -1),
            decoding=True
        )
        inputs_embeds = outputs.inputs_embeds
        attention_mask = outputs.attention_mask
        length = torch.ones_like(input_ids, device=input_ids.device).sum().item() - attention_mask.sum().item()
        length = self.config.max_latent_stage * self.config.c_thought - length
        # get the first token using the current hidden state
        print(attention_mask)
        next_token = torch.argmax(outputs.logits[0, -1]).item()
        tokens.append(next_token)
        new_token_embed = self.embedding(
            torch.tensor(next_token, device=input_ids.device)
        ).view(1, 1, -1)
        new_inputs_embeds = torch.cat((inputs_embeds, new_token_embed), dim=1)
        attention_mask = torch.cat((attention_mask, torch.tensor([[1]], device=input_ids.device)), dim=1)
        position_ids = get_position_ids_from_attention_mask(attention_mask)

        # get other tokens
        for _ in range(max_new_tokens - 1):
            outputs = self.base_causallm(
                inputs_embeds=new_inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids
            )
            self.gen_forward_cnt += 1
            length += 1
            next_token = torch.argmax(outputs.logits[0, -1]).item()
            if next_token == self.eos_token_id:
                break
            tokens.append(next_token)
            new_token_embed = self.embedding(
                torch.tensor(next_token, device=input_ids.device)
            ).view(1, 1, -1)
            new_inputs_embeds = torch.cat((new_inputs_embeds, new_token_embed), dim=1)
            attention_mask = torch.cat((attention_mask, torch.tensor([[1]], device=input_ids.device)), dim=1)
            position_ids = get_position_ids_from_attention_mask(attention_mask)

        if synced_gpus:
            while (
                    self.gen_forward_cnt < max_new_tokens + MAX_N_LATENT
            ):  # leave some room for latent tokens
                self.gen_forward_cnt += 1
                _ = self.base_causallm(
                    inputs_embeds=new_inputs_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids
                )
        if output_embedding:
            # for analysis purpose
            return torch.tensor(tokens).view(1, -1), new_inputs_embeds
        else:
            return torch.tensor(tokens).view(1, -1), length
