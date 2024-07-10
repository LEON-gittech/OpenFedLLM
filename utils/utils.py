import math
import torch
import os
from unsloth import FastLanguageModel 

def cosine_learning_rate(current_round, total_rounds, initial_lr=0.001, min_lr=0):
    """
    Compute the learning rate based on a cosine schedule.

    :param current_round: The current training round (0-indexed).
    :param total_rounds: The total number of training rounds.
    :param initial_lr: The initial learning rate.
    :param min_lr: The minimum learning rate.
    :return: The computed learning rate for the current round.
    """
    # Compute the cosine learning rate
    cosine_lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * current_round / total_rounds))
    return cosine_lr

def is_adapter_checkpoint(path):
    if not os.path.exists(path) or "adapter_model.safetensors" not in os.listdir(path): return False
    else: return True

def get_unsloth_model(model_args):
    max_seq_length = 2048
    if torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
    dtype=None

    if not is_adapter_checkpoint(model_args.model_name_or_path):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_args.model_name_or_path,
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = True,
        )
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_args.model_name_or_path,
            dtype = dtype,
            load_in_4bit=True
        )
    # Do model patching and add fast LoRA weights
    if not is_adapter_checkpoint(model_args.model_name_or_path):
        # model.enable_input_require_grads()
        model = FastLanguageModel.get_peft_model(
            model,
            r = 16,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",],
            lora_alpha = 16,
            lora_dropout = 0, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            random_state = 3407,
            max_seq_length = max_seq_length,
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
        )
    return model, tokenizer




if __name__ == "__main__":

    # Example usage:
    num_rounds = 300
    initial_lr = 5e-5
    min_lr = 1e-6

    lr_list = []
    for round in range(num_rounds):
        lr = cosine_learning_rate(round, num_rounds, initial_lr, min_lr)
        lr_list.append(lr)
        print(f"Round {round + 1}/{num_rounds}, Learning Rate: {lr:.8f}")
