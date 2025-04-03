import logging
import os
from transformers import TextStreamer
import pandas as pd
from typing import List, Dict, Any, Union, Optional
from transformers import PreTrainedTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams

from src.utils.logging.logging_config import setup_logging

setup_logging()
logger = logging.getLogger()


def trim_completion(completion: str) -> str:
    """
    Trims the completion to remove any trailing newlines.

    Args:
        completion: The completion to trim.

    Returns:
        The trimmed completion.
    """
    try:
        if "\n" in completion:
            last_newline = completion.rfind("\n")
            completion = completion[:last_newline]
            return completion.strip()
        else:
            return completion
    except Exception as e:
        logger.error(f"Error in trim_completion: {e}")
        return ""


def extract_completion(answer: str) -> str:
    """
    Extracts the completion from the answer.

    Args:
        answer: The answer to extract the completion from.

    Returns:
        The extracted completion.
    """
    try:
        # pattern = f"[/INST]"
        pattern = "### Response:"
        parts = answer.split(pattern)
        if len(parts) > 1:
            return parts[-1]
        else:
            return ""
    except Exception as e:
        logger.error(f"Error in extract_completion: {e}")
        return ""

template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{} 

### Response:"""

def do_sample(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt: str
) -> str:
    """
    Samples from the model using the prompt.

    Args:
        model: The model to sample from.
        tokenizer: The tokenizer to use.
        prompt: The prompt to use.

    Returns:
        The sampled text.
    """
    try:
        # prompt_sample = [{"role": "user", "content": prompt}]
        # model_prompt = tokenizer.apply_chat_template(prompt_sample, tokenize=False)
        # model_inputs = tokenizer(model_prompt, return_tensors="pt").to("cuda")
        prompt = template.format(prompt)
        model_inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        # streamer = TextStreamer(tokenizer)

        generated_ids = model.generate(
            **model_inputs,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            top_p=0.9,
            temperature=0.6,
            max_new_tokens=256,
            # streamer=streamer,
        )

        answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return answer
    except Exception as e:
        logger.error(f"Error in do_sample: {e}")
        return ""


def generate(
    model: Union[PreTrainedModel, 'LLM'],
    tokenizer: Optional[PreTrainedTokenizer],
    gen_prompts: pd.DataFrame,
    responses_to_generate: int,
    output_path: str = None,
    vllm: int = 0
) -> None:
    """
    为给定提示生成回复并保存到输出路径。

    Args:
        model: 要使用的模型（可以是Transformers模型或vLLM模型）
        tokenizer: 分词器（仅用于Transformers模型）
        gen_prompts: 要生成回复的提示
        responses_to_generate: 每个提示要生成的回复数量
        output_path: 保存生成回复的路径
        vllm: 是否使用vLLM（0：使用Transformers，1：使用vLLM）
    """
    completions = []
    try:
        if vllm:
            # 准备批量生成的提示
            prompts_batch = []
            prompt_ids = []
            for _, row in gen_prompts.iterrows():
                prompt = row["prompt"]
                prompt_id = row["id"]
                # 每个提示生成多个回复
                for _ in range(responses_to_generate):
                    prompts_batch.append(template.format(prompt))
                    prompt_ids.append(prompt_id)
            
            # 使用vLLM批量生成
            sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_tokens=256
            )
            if hasattr(model, "fast_generate"):
                print("generating response...")
                outputs = model.fast_generate(
                    prompts_batch,
                    sampling_params = sampling_params,
                    lora_request = model.load_lora("/mnt/bn/merlin-datavolume-tsy/leon/FedSPA/checkpoints_cache/saved_model_lora"),
                )
            else:
                outputs = model.generate(prompts_batch, sampling_params)
            
            # 处理生成的结果
            for output, prompt_id, prompt in zip(outputs, prompt_ids, prompts_batch):
                completion = output.outputs[0].text
                trimmed_completion = trim_completion(completion)
                
                completions.append({
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "completion": trimmed_completion,
                })
        
        else:
            # 原有的逐个生成逻辑
            for _, row in gen_prompts.iterrows():
                prompt = row["prompt"]
                prompt_id = row["id"]

                for completion_sample in range(responses_to_generate):
                    answer = do_sample(model, tokenizer, prompt)
                    completion = extract_completion(answer)
                    # print(f"completion {completion}\n\n")
                    trimmed_completion = trim_completion(completion)

                    completions.append({
                        "prompt_id": prompt_id,
                        "prompt": prompt,
                        "completion": trimmed_completion,
                    })
        
        # 最后保存一次所有结果
        df_completions = pd.DataFrame(completions)
        df_completions.to_json(output_path, orient="records", lines=True)
        
    except Exception as e:
        logger.error(f"Error in generate: {e}")


def generate_responses(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: Dict[str, Any],
    iteration: int,
    prompts_path: str,
) -> Union[str, None]:
    """
    Generates responses for the given iteration and saves them to the output path.

    Args:
        model: The model to use.
        tokenizer: The tokenizer to use.
        config: The configuration dictionary.
        iteration: The current iteration.
        prompts_path: The path to the prompts.

    Returns:
        The output path where the responses were saved, or None if an error occurred.
    """
    try:
        logger.info(f"Generating responses for iteration {iteration}")
        output_dir = config["data_path"] / f"{iteration}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = output_dir / "gen_responses.jsonl"
        logger.info(f"Output path: {output_path}")

        gen_prompts = pd.read_json(prompts_path, lines=True)
        generate(
            model=model,
            tokenizer=tokenizer,
            gen_prompts=gen_prompts,
            responses_to_generate=config["response_prompts"]["new_prompts"],
            output_path=output_path,
        )
        return output_path
    except Exception as e:
        logger.error(f"Error in generate_responses: {e}")
