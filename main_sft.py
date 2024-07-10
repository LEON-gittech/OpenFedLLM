import copy
import os
from tqdm import tqdm
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training
from datasets import load_from_disk
from utils.utils import get_unsloth_model

from utils import *
# from utils.template import formatting_prompts_func
from federated_learning import *
from config import get_config, save_config, get_model_config, get_training_args
from utils.dataset_utils import *
os.environ["TOKENIZERS_PARALLELISM"]="false"

# ===== Define the arguments =====
script_args, fed_args, peft_config = get_config()
training_args = get_training_args(script_args, script_args.learning_rate)
save_config(script_args, fed_args)
print(script_args, fed_args)

# ===== Load the dataset =====
# dataset = get_dataset(script_args.dataset_name, script_args.local_data_dir)
# dataset = process_sft_dataset(script_args.dataset_name, dataset, script_args.dataset_sample) #对数据集做一些预处理

# # ===== Split the dataset into clients =====
# local_datasets = split_dataset(fed_args, script_args, dataset) #分给不同的客户端，目前只实现了 iid 分布
local_datasets=[]
for i in range(fed_args.num_clients):
    local_datasets.append(load_from_disk(f"/mnt/bn/data-tns-live-llm/leon/datasets/fed_data/pos_{i}.parquet"))
sample_num_list = [len(local_datasets[i]) for i in range(fed_args.num_clients)]

# ===== Get model config =====
device_map, quantization_config, torch_dtype = get_model_config(script_args)

if script_args.unsloth:
    model, tokenizer = get_unsloth_model(script_args)
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
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False, padding_side="right")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token   # following vicuna
# print(tokenizer.eos_token)
# ===== Define the formatting function (cater to TRL SFTTrainer)=====
# response_template = get_formatting_prompts_func(script_args.template, tokenizer.eos_token) #只有'alpaca'和'vicuna' 
formatting_prompts_func, response_template = get_formatting_prompts_func(script_args.template, tokenizer.eos_token) #只有'alpaca'和'vicuna' 
# template， 返回一个函数，用于对输入进行预处理， response_template='\n### Response:' or ' ASSISTANT:'
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]   # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]` for Llama2
data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

# ===== Start federated training =====
training_loss = [[] for i in range(fed_args.num_clients)]

for round in (range(fed_args.num_rounds)):

    clients_this_round = get_clients_this_round(fed_args, round) #随机采样得到

    print(f">> ==================== Round {round+1} : {clients_this_round} ====================")
    
    for client in range(fed_args.num_clients):

        if client not in clients_this_round:
            training_loss[client].append(-1)            # -1 is an indicator of not training
            continue

        set_peft_model_state_dict(model, global_dict)   # sync the global model to the local model，更新本地模型的 lora 参数

        sub_dataset = get_dataset_this_round(local_datasets[client], round, fed_args, script_args)      # get the required sub-dataset for this round， 随机采样，num2sample = script_args.batch_size * script_args.gradient_accumulation_steps * script_args.max_steps
        # print("instruction", len(sub_dataset["instruction"]))
        # print("response", len(sub_dataset["response"]))
        data_module = make_supervised_data_module(tokenizer=tokenizer, dataset=sub_dataset)
        new_lr = cosine_learning_rate(round, fed_args.num_rounds, script_args.learning_rate, 1e-6)      # manually schedule the learning rate
        training_args = get_training_args(script_args, new_lr)

        # ===== Train local model on the client side =====
        trainer = get_fed_local_sft_trainer( #根据不同的算法使用不同的 Trainer
            model=model,
            tokenizer=tokenizer,
            training_args=training_args,
            local_dataset=sub_dataset,
            data_module=data_module,
            formatting_prompts_func=formatting_prompts_func,
            data_collator=data_collator,
            global_dict=global_dict,
            fed_args=fed_args,
            script_args=script_args,
            local_auxiliary=auxiliary_model_list[client],
            global_auxiliary=global_auxiliary,
        )

        results = trainer.train()
        training_loss[client].append(results.training_loss)

        # ===== Client transmits local information to server =====
        if fed_args.fed_alg == 'scaffold':
            auxiliary_model_list[client], auxiliary_delta_dict[client] = trainer.get_auxiliary_param()

        local_dict_list[client] = copy.deepcopy(get_peft_model_state_dict(model))   # copy is needed!

    # ===== Server aggregates the local models =====
    global_dict, global_auxiliary = global_aggregate(
        fed_args, global_dict, local_dict_list, sample_num_list, \
        clients_this_round, round, proxy_dict=proxy_dict, \
        opt_proxy_dict=opt_proxy_dict, auxiliary_info=(global_auxiliary, auxiliary_delta_dict)
    )
    set_peft_model_state_dict(model, global_dict)   # Update global model

    # ===== Save the model =====
    if (round+1) % fed_args.save_model_freq == 0:
        trainer.save_model(os.path.join(script_args.output_dir, f"checkpoint-{round+1}"))
    
    np.save(os.path.join(script_args.output_dir, "training_loss.npy"), np.array(training_loss))

trainer.save_state()
trainer.save_model(output_dir=script_args.output_dir)