import os
import sys
import gc
import json
import yaml
import argparse
import functools
from contextlib import contextmanager
import math
import torch
import torch.optim as optim
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from methods.coconut import Coconut
from methods.sim_cot import CoconutGPT_Same_Word_Embedding
from methods.sim_cot_end import CoconutGPT_Same_Word_Embedding_EndSignal
from methods.sim_cot_vae_end import CoconutGPT_Same_Word_Embedding_EndSignal_VAE
from methods.sim_cot_vae import CoconutGPT_Same_Word_Embedding_VAE
from methods.sim_cot_rl import Coconut_RL
from methods.sim_cot_rl_end import Coconut_RL_End
from methods.sim_cot_rl_vae_end import Coconut_RL_End_VAE
from methods.sim_cot_rl_vae import Coconut_RL_VAE
from methods.sim_cot_rl_end_pentext import Coconut_RL_End_Penalty_on_Conherence
from methods.sim_cot_rl_test_time import Coconut_RL_Test_Time
from methods.sim_cot_rl_test_time_vae import Coconut_RL_Test_Time_VAE
from dataset import (
    get_dataset,
    get_question_latent_dataset,
    get_cot_with_explainable_latent_dataset,
    MyCollator,
    MyExplainableCollator,
)
from utils import Config, set_seed

# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------
import traceback

RUN_EVAL_ON_RANK0 = True


def run_on_rank0(fn, *args, **kwargs):
    dist.barrier()
    result_list = [None]
    error_info = [None]
    rank = dist.get_rank()

    if rank == 0:
        try:
            result_list[0] = fn(*args, **kwargs)
        except Exception as e:
            error_info[0] = (repr(e), traceback.format_exc())

    dist.broadcast_object_list(result_list, src=0)
    dist.broadcast_object_list(error_info, src=0)
    dist.barrier()

    if error_info[0] is not None:
        err_repr, err_tb = error_info[0]
        raise RuntimeError(f"run_on_rank0 exception: {err_repr}\n{err_tb}")

    return result_list[0]


def check_requires_grad(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name} requires gradient")


def save_jsonl_line(filepath, data):
    if not isinstance(data, dict):
        raise ValueError("CODI must be dict")
    with open(filepath, "a", encoding="utf-8") as f:
        json_line = json.dumps(data, ensure_ascii=False)
        f.write(json_line + "\n")


def log_scalar_dict(writer, log_dict, default_step):
    if writer is None:
        return
    step = log_dict.get("train/step", log_dict.get("eval/step", default_step))
    for key, value in log_dict.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(key, value, step)


@contextmanager
def ddp_inference_context(model):
    """
    在 DDP 包裹下拿到底层 module，未包裹时直接返回自身。
    """
    if isinstance(model, DDP):
        yield model.module
    else:
        yield model


def save_ddp_checkpoint(model, save_dir, tag):
    """
    保存 checkpoint（full state dict），只在 rank0 写盘。
    """
    module = model.module if isinstance(model, DDP) else model
    state_dict = module.state_dict()
    if dist.get_rank() == 0:
        ckpt_path = os.path.join(save_dir, f"checkpoint_{tag}")
        torch.save(state_dict, ckpt_path)
        print(f"[rank0] Saved checkpoint -> {ckpt_path}")
    dist.barrier()
    del state_dict
    gc.collect()
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="coconut")
    parser.add_argument("config_file")
    args = parser.parse_args()

    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    with open(args.config_file) as f:
        config_dict = yaml.safe_load(f)

    if rank == 0:
        print("Config:", config_dict)

    configs = Config(config_dict)
    set_seed(configs.seed)
    save_dir = os.path.join(configs.save_path, configs.name)
    os.makedirs(save_dir, exist_ok=True)

    cur_ckpts = os.listdir(save_dir)

    # resume detection
    if len(cur_ckpts) > 0 and not configs.only_eval:
        if rank == 0:
            print(
                "Warning: found previous run and gonna resume from that. "
                "the inputted `resume` argument is ignored!"
            )
        checkpoints = [f for f in cur_ckpts if f.startswith("checkpoint_")]
        checkpoints.sort(key=lambda x: int(x.split("_")[1]))
        latest_checkpoint = checkpoints[-1] if checkpoints else None
        configs.resume = int(latest_checkpoint.split("_")[1])
        load_dir = os.path.join(configs.save_path, configs.name, latest_checkpoint)
        configs.load_model_path = load_dir
        print(f"Loading from previous run epoch_{configs.resume}!")

    elif configs.resume != 0:
        if configs.load_model_path == "None":
            print(
                "Warning: you want to skip the first "
                f"{configs.resume} but not loading checkpoint!"
            )
        print(
            f"Loading from {configs.load_model_path} "
            f"and skip the first {configs.resume} epochs"
        )

    # -----------------------
    # build model(s)
    # -----------------------
    model = AutoModelForCausalLM.from_pretrained(configs.model_id).to(local_rank)

    need_explainable = configs.mode != "coconut_baseline"
    explainable_model = None
    if need_explainable:
        explainable_model = AutoModelForCausalLM.from_pretrained(
            configs.model_id
        ).to(local_rank)

    tokenizer = AutoTokenizer.from_pretrained(configs.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    is_rl_mode = getattr(configs, "rl_training",
                         False) or configs.mode == "coconut_rl" or configs.mode == "coconut_rl_end" or configs.mode == "coconut_rl_end_vae" or configs.mode == "coconut_rl_end_pentext" or configs.mode == "coconut_rl_test_time" or configs.mode == "coconut_rl_test_time_vae" or configs.mode == "coconut_rl_vae"

    # load pretrained checkpoint before wrapping
    loaded = False
    if configs.load_model_path != "None":
        saved_weights = torch.load(configs.load_model_path, map_location="cpu")

        if configs.coconut and not any(
                k.startswith("base_causallm") for k in saved_weights.keys()
        ):
            loaded = True
            print(model.load_state_dict(saved_weights, strict=False))

        elif not configs.coconut and any(
                k.startswith("base_causallm") for k in saved_weights.keys()
        ):
            raise ValueError("Cannot load coconut model weights into a causallm model")

        elif configs.coconut and any(
                k.startswith("base_causallm") for k in saved_weights.keys()
        ):
            pass
        else:
            loaded = True
            print(model.load_state_dict(saved_weights, strict=False))

    if not (configs.cot or configs.no_thoughts or configs.no_cot):
        model.resize_token_embeddings(len(tokenizer))
        embeddings = model.get_input_embeddings()
        target_id = tokenizer.convert_tokens_to_ids("<<")
        for token_id in [latent_id, start_id, end_id]:
            target_embedding = embeddings.weight.data[target_id]
            embeddings.weight.data[token_id] = target_embedding
            if hasattr(model, "lm_head"):
                model.lm_head.weight.data[token_id] = model.lm_head.weight.data[target_id]

    if configs.no_thoughts:
        configs.c_thought = 0
        configs.coconut = False

    if configs.coconut:
        if configs.mode == "coconutgpt_same_word_embedding":
            model = CoconutGPT_Same_Word_Embedding(
                model,
                explainable_model,
                tokenizer,
                latent_id,
                start_id,
                end_id,
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<<"),
                configs.c_thought,
                configs,
            )
        elif configs.mode == "coconutgpt_same_word_embedding_vae":
            model = CoconutGPT_Same_Word_Embedding_VAE(
                model,
                explainable_model,
                tokenizer,
                latent_id,
                start_id,
                end_id,
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<<"),
                configs.c_thought,
                configs,
            )
        elif configs.mode == "coconutgpt_same_word_embedding_len":
            model = CoconutGPT_Same_Word_Embedding_EndSignal(
                model,
                explainable_model,
                tokenizer,
                latent_id,
                start_id,
                end_id,
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<<"),
                configs.c_thought,
                configs,
            )
        elif configs.mode == "coconutgpt_same_word_embedding_len_vae":
            model = CoconutGPT_Same_Word_Embedding_EndSignal_VAE(
                model,
                explainable_model,
                tokenizer,
                latent_id,
                start_id,
                end_id,
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<<"),
                configs.c_thought,
                configs,
            )
        elif configs.mode == "coconut_baseline":
            model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id)
        elif configs.mode == "coconut_rl":
            model = Coconut_RL(
                model,
                explainable_model,
                latent_id,
                start_id,
                end_id,
                tokenizer.eos_token_id,
                tokenizer=tokenizer,
                rl_config=configs,
            )
        elif configs.mode == "coconut_rl_end":
            model = Coconut_RL_End(
                model,
                explainable_model,
                latent_id,
                start_id,
                end_id,
                tokenizer.eos_token_id,
                tokenizer=tokenizer,
                rl_config=configs,
            )
        elif configs.mode == "coconut_rl_end_vae":
            model = Coconut_RL_End_VAE(
                model,
                explainable_model,
                latent_id,
                start_id,
                end_id,
                tokenizer.eos_token_id,
                tokenizer=tokenizer,
                rl_config=configs,
            )
        elif configs.mode == "coconut_rl_end_pentext":
            model = Coconut_RL_End_Penalty_on_Conherence(
                model,
                explainable_model,
                latent_id,
                start_id,
                end_id,
                tokenizer.eos_token_id,
                tokenizer=tokenizer,
                rl_config=configs,
            )
        elif configs.mode == "coconut_rl_vae":
            model = Coconut_RL_VAE(
                model,
                explainable_model,
                latent_id,
                start_id,
                end_id,
                tokenizer.eos_token_id,
                tokenizer=tokenizer,
                rl_config=configs,
            )
        elif configs.mode == "coconut_rl_test_time_vae":
            model = Coconut_RL_Test_Time_VAE(
                model,
                explainable_model,
                latent_id,
                start_id,
                end_id,
                tokenizer.eos_token_id,
                tokenizer=tokenizer,
                rl_config=configs,
            )
        elif configs.mode == "coconut_rl_test_time":
            model = Coconut_RL_Test_Time(
                model,
                explainable_model,
                latent_id,
                start_id,
                end_id,
                tokenizer.eos_token_id,
                tokenizer=tokenizer,
                rl_config=configs,
            )
        else:
            raise ValueError(f"don't support model {configs.mode=}")

    if configs.load_model_path != "None" and not loaded:
        print(model.load_state_dict(saved_weights, strict=False))

    print(f"Running DDP on rank = {rank}, world size = {world_size}")
    model = model.to(local_rank)
    dist.barrier()

    if configs.bf16:
        model.to(torch.bfloat16)

    if is_rl_mode and configs.only_eval:
        model.is_training = False
        parallel_model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
    elif configs.only_eval:
        parallel_model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
        del model
    elif configs.mode == "coconutgpt_same_word_embedding_len":
        parallel_model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
        del model
    else:
        parallel_model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
        del model

    if rank == 0:
        print(parallel_model)
    check_requires_grad(
        parallel_model.module if isinstance(parallel_model, DDP) else parallel_model
    )

    # -----------------------
    # dataset / tokenizer etc
    # -----------------------
    question_val = None
    answers_val = None
    cot_val = None

    base_dataset_valid = get_dataset(
        configs.val_path, tokenizer, max_size=32 if configs.debug else 10 ** 9
    )
    base_dataset_test = None
    if hasattr(configs, "test_path"):
        base_dataset_test = get_dataset(
            configs.test_path, tokenizer, max_size=32 if configs.debug else 10 ** 9
        )
        question_val = [d["question"] for d in json.load(open(configs.test_path))]
        answers_val = [
            d["answer"].replace(",", "").strip() for d in json.load(open(configs.test_path))
        ]
        cot_val = ["\n".join(d["steps"]) for d in json.load(open(configs.test_path))]
    else:
        base_dataset_test = base_dataset_valid
    if not configs.only_eval or is_rl_mode:
        base_dataset_train = get_dataset(
            configs.train_path, tokenizer, max_size=32 if configs.debug else 10 ** 9
        )

    max_new_tokens = 16

    total_train_steps = 0
    writer = None
    if (
            not configs.debug
            and not configs.only_eval
            and configs.logger
            and rank == 0
    ):
        tb_log_dir = os.path.join(save_dir, "tensorboard_logs")
        os.makedirs(tb_log_dir, exist_ok=True)
        writer = SummaryWriter(tb_log_dir)
        print(f"TensorBoard logging dir: {tb_log_dir}")

    best_acc = 0

    eval_collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)
    if not configs.only_eval or is_rl_mode:
        train_collator = MyExplainableCollator(tokenizer, latent_id=latent_id)
    else:
        train_collator = eval_collator

    # -----------------------
    # epoch loop
    # -----------------------

    for epoch in range(0 if is_rl_mode else configs.resume, configs.num_epochs):
        scheduled_stage = (
            0
            if (configs.cot or configs.no_cot)
            else (epoch // configs.epochs_per_stage + 1)
        )
        dataset_gen_val = get_question_latent_dataset(
            10000,
            base_dataset_test,
            configs,
            start_id,
            latent_id,
            end_id,
            no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
        )

        eval_sampler = (
            None if RUN_EVAL_ON_RANK0 else DistributedSampler(dataset_gen_val, shuffle=False)
        )
        valid_gen_dataloader = torch.utils.data.DataLoader(
            dataset_gen_val,
            num_workers=1,
            pin_memory=True,
            batch_size=1,
            collate_fn=train_collator,
            sampler=eval_sampler,
            shuffle=False if eval_sampler is None else None,
        )

        # =====================================================
        # RL training branch
        # =====================================================
        if is_rl_mode:
            dataset_loss_val = get_cot_with_explainable_latent_dataset(
                10000,
                base_dataset_valid,
                configs,
                start_id,
                latent_id,
                end_id,
                no_special_marker=(
                        configs.cot or configs.no_cot or configs.no_thoughts
                ),
            )
            eval_sampler = (
                None
                if RUN_EVAL_ON_RANK0
                else DistributedSampler(dataset_loss_val, shuffle=False)
            )
            valid_loss_dataloader = torch.utils.data.DataLoader(
                dataset_loss_val,
                num_workers=1,
                pin_memory=True,
                batch_size=configs.batch_size_validating,
                collate_fn=train_collator,
                sampler=eval_sampler,
                shuffle=False if eval_sampler is None else None,
            )

            dataset_train = get_cot_with_explainable_latent_dataset(
                10000,
                base_dataset_train,
                configs,
                start_id,
                latent_id,
                end_id,
                no_special_marker=(
                        configs.cot or configs.no_cot or configs.no_thoughts
                ),
                shuffle=True,
            )
            train_dataloader = torch.utils.data.DataLoader(
                dataset_train,
                num_workers=1,
                shuffle=True,
                pin_memory=True,
                batch_size=configs.batch_size_training,
                collate_fn=train_collator,
            )

            optimizer = optim.AdamW(
                parallel_model.parameters(),
                lr=configs.lr,
                weight_decay=configs.weight_decay,
            )

            parallel_model.train()
            total_length = len(train_dataloader)
            pbar = tqdm(
                colour="blue",
                desc=f"RL Epoch: {epoch + 1}",
                total=total_length,
                dynamic_ncols=True,
            )

            for step, batch in enumerate(train_dataloader):
                if step < configs.resume:
                    continue
                if total_train_steps % configs.steps_validating == 0:
                    def _eval_fn():
                        print(
                            f"[Epoch {epoch + 1}] Test running.........................."
                        )
                        with ddp_inference_context(parallel_model) as eval_model:
                            return eval_model.eval_generation(
                                valid_loss_dataloader, "val"
                            )

                    log_dict = run_on_rank0(_eval_fn)
                    parallel_model.train()
                    if rank == 0 and writer and log_dict:
                        for key, value in log_dict.items():
                            if isinstance(value, (int, float)):
                                writer.add_scalar(key, value, total_train_steps)

                if configs.only_eval:
                    break
                log_dict = parallel_model.module.rl_step(batch, optimizer, writer) \
                    if isinstance(parallel_model, DDP) else parallel_model.rl_step(batch, optimizer, writer)

                if rank == 0 and writer and log_dict:
                    step_idx = log_dict.get("train/step", total_train_steps)
                    for key, value in log_dict.items():
                        if isinstance(value, (int, float)):
                            writer.add_scalar(key, value, step_idx)

                pbar.update(1)
                reward = log_dict.get("train/rewards", 0.0) if log_dict else 0.0
                pbar.set_description(
                    f"RL Epoch {epoch + 1}/{configs.num_epochs} | "
                    f"step {step}/{len(train_dataloader)} | reward {reward:.4f}"
                )

                if (total_train_steps + 1) % configs.save_every_k == 0:
                    save_ddp_checkpoint(
                        parallel_model, save_dir, total_train_steps + 1
                    )

                total_train_steps += 1

            pbar.close()

            if (
                    configs.save_ckpts
                    and not configs.save_only_improve
                    and not configs.debug
                    and not configs.only_eval
            ):
                save_ddp_checkpoint(parallel_model, save_dir, epoch + 1)

        # =====================================================
        # Supervised branch
        # =====================================================
        else:

            # ===========================================================
            # Generation evaluation
            # ===========================================================
            def _generation_eval():
                print(
                    f"[Epoch {epoch + 1}] Test running.........................."
                )
                with ddp_inference_context(parallel_model) as eval_model:
                    eval_model.eval()
                    cor = torch.tensor(0, device=local_rank)
                    cor_cot = torch.tensor(0, device=local_rank)
                    total = torch.tensor(0, device=local_rank)
                    pbar = tqdm(
                        colour="blue",
                        desc="Test Accuracy",
                        total=len(valid_gen_dataloader),
                        dynamic_ncols=True,
                    )
                    with torch.no_grad():
                        for idx, batch in enumerate(valid_gen_dataloader):
                            batch_gpu = {
                                k: v.to(local_rank)
                                for k, v in batch.items()
                                if v is not None and k not in ["idx", "position_ids"]
                            }
                            test_idx = batch["idx"][0]
                            answer = answers_val[test_idx.item()]
                            answer_cot = cot_val[test_idx.item()]
                            total += 1

                            outputs, length = eval_model.generate(
                                **batch_gpu,
                                max_new_tokens=max_new_tokens,
                            )
                            text_output = tokenizer.decode(
                                outputs[0], skip_special_tokens=True
                            )
                            answer_output = (
                                text_output.split("<|end-latent|>###")[-1]
                                .replace(",", "")
                                .strip()
                            )
                            if idx < 5 or configs.only_eval:
                                print(
                                    f"[Sample {idx}] GT answer = '{answer}', predicted = '{answer_output}'"
                                )
                            try:  # some answers may be like '10.0' but predicted as '10'
                                answer_output = float(answer_output)
                                answer = float(answer)
                                cor += (abs(answer_output - answer) < 0.05)
                            except ValueError:
                                cor += answer_output == answer
                            cor_cot += length
                            pbar.update(1)
                            pbar.set_description(
                                f"Test acc: {round(float(cor / total * 100), 2)}%"
                            )
                    pbar.close()
                    return cor, cor_cot, total

            test_result = run_on_rank0(_generation_eval) if RUN_EVAL_ON_RANK0 else _generation_eval()
            if rank == 0 and test_result is not None:
                cor, cor_cot, total = test_result
                cor = cor.item();
                cor_cot = cor_cot.item();
                total = total.item()
                acc = cor / max(total, 1)
                cot_em = cor_cot / max(total, 1)
                print(
                    f"[Epoch {epoch + 1}] Test accuracy: {cor}/{total}={acc:.4f}, "
                    f"Answer-part-length: {cor_cot}/{total}={cot_em:.4f}"
                )
                if writer:
                    writer.add_scalar("test/acc", acc, epoch + 1)
                    writer.add_scalar("test/cot_em", cot_em, epoch + 1)

            if configs.only_eval:
                break

            dataset_loss_val = get_cot_with_explainable_latent_dataset(
                scheduled_stage,
                base_dataset_valid,
                configs,
                start_id,
                latent_id,
                end_id,
                no_special_marker=(
                        configs.cot or configs.no_cot or configs.no_thoughts
                ),
                shuffle=False,
            )
            valid_loss_dataloader = torch.utils.data.DataLoader(
                dataset_loss_val,
                num_workers=1,
                shuffle=False,
                pin_memory=True,
                batch_size=configs.batch_size_validating,
                collate_fn=train_collator,
                sampler=DistributedSampler(dataset_loss_val, shuffle=False),
            )

            if (
                    configs.save_ckpts
                    and not configs.save_only_improve
                    and not configs.debug
                    and not configs.only_eval
            ):
                save_ddp_checkpoint(parallel_model, save_dir, epoch)

            def _supervised_eval():
                total_loss = 0.0
                total_loss_explain_all = 0.0
                with ddp_inference_context(parallel_model) as eval_model:
                    eval_model.eval()
                    with torch.no_grad():
                        for batch in valid_loss_dataloader:
                            batch = {
                                key: batch[key].to(local_rank)
                                for key in batch.keys()
                                if key != "idx"
                            }
                            outputs = eval_model(**batch)
                            total_loss += outputs.loss.item()
                            total_loss_explain_all += (
                                outputs.loss_explain_all.item()
                            )
                return (
                    total_loss / max(len(valid_loss_dataloader), 1),
                    total_loss_explain_all / max(len(valid_loss_dataloader), 1),
                )

            if RUN_EVAL_ON_RANK0:
                eval_loss, eval_loss_explain = run_on_rank0(_supervised_eval)
            else:
                eval_loss, eval_loss_explain = _supervised_eval()

            parallel_model.train()

            if rank == 0 and writer:
                writer.add_scalar(
                    "eval/loss_explain_all", eval_loss_explain, epoch + 1
                )
                writer.add_scalar("eval/loss", eval_loss, epoch + 1)
                print("eval loss", eval_loss)

            dataset_train = get_cot_with_explainable_latent_dataset(
                scheduled_stage,
                base_dataset_train,
                configs,
                start_id,
                latent_id,
                end_id,
                no_special_marker=(
                        configs.cot or configs.no_cot or configs.no_thoughts
                ),
                shuffle=True,
            )

            train_dataloader = torch.utils.data.DataLoader(
                dataset_train,
                num_workers=1,
                shuffle=False,
                pin_memory=True,
                batch_size=configs.batch_size_training,
                collate_fn=train_collator,
                sampler=DistributedSampler(dataset_train, shuffle=True),
            )

            optimizer = optim.AdamW(
                parallel_model.parameters(),
                lr=configs.lr,
                weight_decay=configs.weight_decay,
            )

            parallel_model.train()

            total_length = len(train_dataloader) // configs.gradient_accumulation_steps
            pbar = tqdm(
                colour="blue",
                desc=f"Training Epoch: {epoch + 1}",
                total=total_length,
                dynamic_ncols=True,
            )

            for step, batch in enumerate(train_dataloader):
                if step == 0 and writer and rank == 0:
                    cur_bs = len(batch["input_ids"])
                    text_str = ""
                    for data_idx in range(cur_bs):
                        for token_idx in range(len(batch["input_ids"][data_idx])):
                            text_str += (
                                f"{batch['input_ids'][data_idx][token_idx].item()} "
                                f"{batch['labels'][data_idx][token_idx].item()} "
                                f"{tokenizer.decode(batch['input_ids'][data_idx][token_idx])}\n"
                            )
                        text_str += "====" * 10 + "\n"
                    writer.add_text(
                        "train/sample_batch", text_str, total_train_steps
                    )

                total_train_steps += 1
                batch = {
                    key: batch[key].to(local_rank)
                    for key in batch.keys()
                    if key != "idx"
                }

                outputs = parallel_model(**batch)

                loss = outputs.loss / configs.gradient_accumulation_steps
                loss.backward()

                if ((step + 1) % configs.gradient_accumulation_steps == 0) or (
                        step == len(train_dataloader) - 1
                ):
                    optimizer.step()
                    optimizer.zero_grad()
                    pbar.update(1)

                if rank == 0 and writer:
                    loss_scalar = loss.detach().item() * configs.gradient_accumulation_steps
                    loss_explain_scalar = (
                                                  outputs.loss_explain_all / configs.gradient_accumulation_steps
                                          ).detach().item() * configs.gradient_accumulation_steps
                    # loss_stop = (
                    #                     outputs.loss_stop / configs.gradient_accumulation_steps
                    #             ).detach().item() * configs.gradient_accumulation_steps
                    # writer.add_scalar(
                    #     "train/loss_stop",
                    #     loss_stop,
                    #     epoch * len(train_dataloader) + step,
                    # )
                    writer.add_scalar(
                        "train/loss_explain_all",
                        loss_explain_scalar,
                        epoch * len(train_dataloader) + step,
                    )
                    writer.add_scalar(
                        "train/loss",
                        loss_scalar,
                        epoch * len(train_dataloader) + step,
                    )
                    writer.add_scalar(
                        "train/epoch", epoch + 1, epoch * len(train_dataloader) + step
                    )

                pbar.set_description(
                    f"Training Epoch: {epoch + 1}/{configs.num_epochs}, "
                    f"batch {step}/{len(train_dataloader)} completed "
                    f"(loss: {round(float(loss.detach().float() * configs.gradient_accumulation_steps), 4)})"
                )
            pbar.close()
            dist.barrier()

            if not configs.only_eval and (epoch + 1) % configs.save_every_k == 0:
                save_ddp_checkpoint(parallel_model, save_dir, epoch + 1)

    if writer:
        writer.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
