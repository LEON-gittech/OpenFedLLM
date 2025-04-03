import copy
import os
from tqdm import tqdm
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM, DPOTrainer
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training
from datasets import load_from_disk
from federated_learning.split_dataset import get_dataset_this_round_fewshot, get_dataset_this_round_QA
from utils.utils import get_unsloth_model, get_unsloth_model_new

from utils import *
# from utils.template import formatting_prompts_func
from federated_learning import *
from config import get_config, save_config, get_model_config, get_training_args
from utils.dataset_utils import *
import random
from src.utils.generate_preferences import generate as generate_preferences
from src.utils.generate_prompts import generate as generate_prompts
from src.utils.generate_responses import generate as generate_responses
from src.utils.generate_scores import generate as generate_scores
# from src.utils.intrinsic_generate_scores import generate as generate_scores
from src.utils.read_write_jsonl import read_jsonl_file, write_jsonl_file
from src.utils.generate_dpo_dataset import generate_dpo_dataset
from utils.FedSPA_utils import compute_ranknet_loss, ranknet_loss, compute_spa_loss
from unsloth import FastLanguageModel
import pandas as pd
from vllm import LLM, SamplingParams
import gc
os.environ["TOKENIZERS_PARALLELISM"]="false"

# ===== Define the arguments =====
script_args, fed_args, peft_config = get_config()
training_args = get_training_args(script_args, script_args.learning_rate)
save_config(script_args, fed_args)
print(script_args, fed_args)

# ===== Get model config =====
device_map, quantization_config, torch_dtype = get_model_config(script_args)

if script_args.unsloth:
    print("using unsloth model")
    model, tokenizer = get_unsloth_model_new(script_args)
    if script_args.rank_net:
        ref_model, _ = get_unsloth_model_new(script_args)
else:
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=script_args.trust_remote_code,
        torch_dtype=torch_dtype,
    )

    if script_args.load_in_8bit or script_args.load_in_4bit:
        model = prepare_model_for_kbit_training(
                    model, use_gradient_checkpointing=training_args.gradient_checkpointing
                )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

# ===== Define the global and local models =====
global_dict = copy.deepcopy(get_peft_model_state_dict(model)) #lora参数
local_dict_list = [copy.deepcopy(global_dict) for i in range(fed_args.num_clients)]
proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict) # 'fedadagrad', 'fedyogi', 'fedadam', 'fedavgm'这四个算法会用到
global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict) #'scaffold'会用到

# ===== Define the tokenizer =====
if not script_args.unsloth:
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False, padding_side="right", model_max_length=script_args.seq_length)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token   # following vicuna
# print(tokenizer.eos_token)

# ===== Define the ift data for generating prompts =====
ift_data = read_jsonl_file("/mnt/bn/data-tns-live-llm/leon/Self-Rewarding-Language-Models/M0/train/ift.jsonl")

# ===== Start federated training =====
training_loss = [[] for i in range(fed_args.num_clients)]
print(fed_args.num_rounds)
sample_num_list = [script_args.prompt_num] * fed_args.num_clients
prev_prompts = [[] for i in range(fed_args.num_clients)]
for round in (range(fed_args.num_rounds)):

    clients_this_round = get_clients_this_round(fed_args, round) #随机采样得到

    print(f">> ==================== Round {round+1} : {clients_this_round} ====================")
    
    for client in range(fed_args.num_clients):
        if client not in clients_this_round:
            training_loss[client].append(-1)            # -1 is an indicator of not training
            continue
        if script_args.rank_net: optimizer = torch.optim.AdamW(model.parameters(), lr=script_args.learning_rate)

        new_lr = cosine_learning_rate(round, fed_args.num_rounds, script_args.learning_rate, 1e-6)      # manually schedule the learning rate
        training_args = get_training_args(script_args, new_lr)

        losses = []
        for epoch in range(script_args.self_reward_epochs):

            print(f">> ==================== Epoch {epoch} : {client} ====================")
            set_peft_model_state_dict(model, global_dict)   # sync the global model to the local model，更新本地模型的 lora 参数
            
            
            if epoch == 0 and script_args.rank_net:
                set_peft_model_state_dict(ref_model, global_dict) 
                ref_model.save_lora("/mnt/bn/merlin-datavolume-tsy/leon/FedSPA/checkpoints_cache/saved_ref_model_lora")

            #保存每个epoch最新的lora参数
            model.save_lora("/mnt/bn/merlin-datavolume-tsy/leon/FedSPA/checkpoints_cache/saved_model_lora")

            # ===== Generate new prompts =====
            gen_prompts = generate_prompts(
                model, tokenizer, ift_data, script_args.prompt_num, vllm=script_args.use_vllm, prev_prompts=prev_prompts[client],
            )
            if gen_prompts == []: exit(0)

            # ===== Generating responses =====
            gen_prompts = pd.DataFrame(gen_prompts)
            prev_prompts[client].extend(gen_prompts["prompt"].to_list())
            
            generate_output_dir = training_args.output_dir.split("/")[-1]
            if not os.path.exists(f"{script_args.generate_data_path}/{generate_output_dir}"):
                os.mkdir(f"{script_args.generate_data_path}/{generate_output_dir}")

            output_path = f"{script_args.generate_data_path}/{generate_output_dir}/{client}_{round}_responses.json"
            generate_responses(
                model=model,
                tokenizer=tokenizer,
                gen_prompts=gen_prompts,
                responses_to_generate=script_args.response_num,
                output_path=output_path,
                vllm=script_args.use_vllm
            )

            # ===== Generating scores =====
            gen_responses = pd.read_json(output_path, lines=True)
            output_path = f"{script_args.generate_data_path}/{generate_output_dir}/{client}_{round}_scores.json"
            score_cache_path = f"{script_args.generate_data_path}/{generate_output_dir}/scores_cache.json"
            
            generate_scores(
                model=model,
                tokenizer=tokenizer,
                gen_respones=gen_responses,
                output_path=output_path,
                vllm=script_args.use_vllm
            )

            if epoch !=0 and script_args.rank_net: # 生成ref_model的scores
                # intrinsic_scores, ref_intrinsic_scores = generate_scores(
                #     model=model,
                #     ref_model=ref_model,
                #     tokenizer=tokenizer,
                #     gen_respones=gen_responses,
                #     output_path=output_path,
                #     vllm=script_args.use_vllm
                # )
                generate_scores(
                    model=ref_model,
                    tokenizer=tokenizer,
                    gen_respones=gen_responses,
                    output_path=score_cache_path,
                    vllm=script_args.use_vllm,
                    is_ref=1
                )

                
                # ===== Computing ranknet loss =====
                # intrinsic_scores = intrinsic_scores.view(script_args.prompt_num, 4)
                # ref_intrinsic_scores = ref_intrinsic_scores.view(script_args.prompt_num, 4)
                reg_loss = compute_ranknet_loss(output_path, score_cache_path)
                # reg_loss = compute_spa_loss(output_path, score_cache_path)
                print(f"reg_loss: {reg_loss}")
                optimizer.zero_grad()
                reg_loss.backward()
                optimizer.step()
            
            # ===== Generating preferences =====
            preference_path = f"{script_args.generate_data_path}/{generate_output_dir}/{client}_{round}_preferences.json"
            generate_preferences(scores_path=output_path, output_path=preference_path)

            # ===== Training DPO model =====
            dpo_dataset = generate_dpo_dataset(preference_path, tokenizer)


            trainer = DPOTrainer(
                model=model,
                train_dataset=dpo_dataset,
                max_length=script_args.seq_length,
                max_prompt_length=script_args.seq_length,
                tokenizer=tokenizer,
                args=training_args,
            )

            results = trainer.train()
            losses.append(results.training_loss)

            # 保存client 每轮每个 epoch 的模型用于衡量preference一致性
            model.save_pretrained(f"{training_args.output_dir}/{round}/{client}/{epoch}")
            tokenizer.save_pretrained(f"{training_args.output_dir}/{round}/{client}/{epoch}")

            # ===== Client transmits local information to server =====
            if fed_args.fed_alg == 'scaffold':
                auxiliary_model_list[client], auxiliary_delta_dict[client] = trainer.get_auxiliary_param()

            local_dict_list[client] = copy.deepcopy(get_peft_model_state_dict(model))   # copy is needed!

        training_loss[client].append(np.mean(np.array(losses)))

    # ===== Server aggregates the local models =====
    global_dict, global_auxiliary = global_aggregate(
        fed_args, global_dict, local_dict_list, sample_num_list, \
        clients_this_round, round, proxy_dict=proxy_dict, \
        opt_proxy_dict=opt_proxy_dict, auxiliary_info=(global_auxiliary, auxiliary_delta_dict)
    )
    set_peft_model_state_dict(model, global_dict)   # Update global model

    # ===== Save the model =====
    if (round+1) % fed_args.save_model_freq == 0:
        model.save_pretrained(os.path.join(script_args.output_dir, f"checkpoint-{round+1}"))
        tokenizer.save_pretrained(os.path.join(script_args.output_dir, f"checkpoint-{round+1}"))
        # trainer.save_model(os.path.join(script_args.output_dir, f"checkpoint-{round+1}"))
    
    np.save(os.path.join(script_args.output_dir, "training_loss.npy"), np.array(training_loss))

trainer.save_state()
trainer.save_model(output_dir=script_args.output_dir)