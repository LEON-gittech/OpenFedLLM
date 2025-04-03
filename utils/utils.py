import math
import torch
import os
from unsloth import FastLanguageModel 

# def prepare_model_for_kbit_training(model, use_gradient_checkpointing=True):
#     r"""
#     This method wraps the entire protocol for preparing a model before running a training. This includes:
#         1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
#         head to fp32

#     Args:
#         model, (`transformers.PreTrainedModel`):
#             The loaded model from `transformers`
#     """
#     loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)

#     for name, param in model.named_parameters():
#         # freeze base model's layers
#         param.requires_grad = False

#     # cast all non INT8/INT4 parameters to fp32
#     for param in model.parameters():
#         if ((param.dtype == torch.float16) or (param.dtype == torch.bfloat16)) and loaded_in_kbit:
#             param.data = param.data.to(torch.float32)

#     for name, module in model.named_modules():
#         if 'norm' in name:
#             module = module.to(torch.float32)

#     if loaded_in_kbit and use_gradient_checkpointing:
#         # For backward compatibility
#         if hasattr(model, "enable_input_require_grads"):
#             model.enable_input_require_grads()
#         else:
#             def make_inputs_require_grad(module, _input, output):
#                 output.requires_grad_(True)

#             model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
#         # enable gradient checkpointing for memory efficiency
#         model.gradient_checkpointing_enable()

#     return model

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

def get_unsloth_model(script_args):
    max_seq_length = 2048
    dtype=None

    if not is_adapter_checkpoint(script_args.model_name_or_path):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = script_args.model_name_or_path,
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = True,
        )
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = script_args.model_name_or_path,
            dtype = dtype,
            load_in_4bit=True
        )
    # Do model patching and add fast LoRA weights
    if not is_adapter_checkpoint(script_args.model_name_or_path):
        # model.enable_input_require_grads()
        print("Loading Adapter...")
        model = FastLanguageModel.get_peft_model(
            model,
            r = script_args.peft_lora_r,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",],
            lora_alpha = script_args.peft_lora_alpha,
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

def get_unsloth_model_new(script_args):
    max_seq_length = 2048
    dtype=None

    assert not is_adapter_checkpoint(script_args.model_name_or_path)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = script_args.model_name_or_path,
        max_seq_length = max_seq_length,
        load_in_4bit = True, # False for LoRA 16bit
        fast_inference = True, # Enable vLLM fast inference
        max_lora_rank = script_args.peft_lora_r,
        gpu_memory_utilization = 0.4, # Reduce if out of memory
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = script_args.peft_lora_r, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
        ], # Remove QKVO if out of memory
        lora_alpha = script_args.peft_lora_alpha,
        use_gradient_checkpointing = "unsloth", # Enable long context finetuning
        random_state = 3407,
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
