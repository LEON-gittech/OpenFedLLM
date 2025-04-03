import os
import logging
import re
import pandas as pd
from typing import List, Dict, Any, Union, Optional
from transformers import PreTrainedTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams

from src.utils.prompts import judge_prompt
from src.utils.logging.logging_config import setup_logging

setup_logging()
logger = logging.getLogger()

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
        model_prompt = template.format(prompt)
        model_inputs = tokenizer(model_prompt, return_tensors="pt").to("cuda")

        generated_ids = model.generate(
            **model_inputs,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            top_p=0.9,
            temperature=0.7,
            max_new_tokens=100,
        )

        answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return answer
    except Exception as e:
        logger.error(f"Error in do_sample: {e}")
        return ""


def extract_scores(answer: str) -> int:
    """
    Extracts the score from the answer.

    Args:
        answer: The answer to extract the score from.

    Returns:
        The extracted score.
    """
    try:
        pattern = r"[Ss]core: ([0-5])"
        matches = re.findall(pattern, answer)
        score = int(matches[0]) if matches else -1
        return score
    except Exception as e:
        logger.error(f"Error in extract_scores: {e}")
        return -1


def generate(
    model: Union[PreTrainedModel, 'LLM'],
    tokenizer: Optional[PreTrainedTokenizer],
    gen_respones: pd.DataFrame,
    output_path: str,
    vllm: int = 0,
    is_ref: int = 0
) -> None:
    """
    为给定的回复生成评分并保存到输出路径。

    Args:
        model: 要使用的模型（可以是Transformers模型或vLLM模型）
        tokenizer: 分词器（仅用于Transformers模型）
        gen_respones: 要评分的回复数据
        output_path: 保存评分结果的路径
        vllm: 是否使用vLLM（0：使用Transformers，1：使用vLLM）
    """
    results = []
    try:
        if vllm:
            # 准备批量评分的提示
            prompts_batch = []
            prompt_ids = []
            prompts = []
            completions = []
            
            for _, row in gen_respones.iterrows():
                prompt = row["prompt"]
                prompt_id = row["prompt_id"]
                completion = row["completion"]
                
                formatted_llm_prompt = template.format(judge_prompt.format(
                    prompt=prompt, response=completion
                ))
                prompts_batch.append(formatted_llm_prompt)
                prompt_ids.append(prompt_id)
                prompts.append(prompt)
                completions.append(completion)
            
            # 使用vLLM批量生成评分
            sampling_params = SamplingParams(
                temperature=0.6,
                top_p=0.9,
                max_tokens=256
            )
            # print(prompts_batch)
            if hasattr(model, "fast_generate"):
                if is_ref:
                    lora_path = "/mnt/bn/merlin-datavolume-tsy/leon/FedSPA/checkpoints_cache/saved_ref_model_lora"
                else:
                    lora_path = "/mnt/bn/merlin-datavolume-tsy/leon/FedSPA/checkpoints_cache/saved_model_lora"

                outputs = model.fast_generate(
                    prompts_batch,
                    sampling_params = sampling_params,
                    lora_request = model.load_lora(lora_path),
                )
            else:
                outputs = model.generate(prompts_batch, sampling_params)
            
            # 处理生成的结果
            scores = []
            for output, prompt_id, prompt, completion in zip(outputs, prompt_ids, prompts, completions):
                answer = output.outputs[0].text
                # print(answer)
                score = extract_scores(answer)
                scores.append(score)
                results.append({
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "completion": completion,
                    "score": score,
                    "reasoning": answer,
                })
                # print(f"Scores {score}")
        else:
            # 原有的逐个评分逻辑
            for _, row in gen_respones.iterrows():
                prompt = row["prompt"]
                prompt_id = row["prompt_id"]
                completion = row["completion"]

                formatted_llm_prompt = judge_prompt.format(
                    prompt=prompt, response=completion
                )

                answer = do_sample(model, tokenizer, formatted_llm_prompt)
                score = extract_scores(answer)
                results.append({
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "completion": completion,
                    "score": score,
                    "reasoning": answer,
                })

        # 保存所有结果
        df_results = pd.DataFrame(results)
        df_results.to_json(output_path, orient="records", lines=True)
        
    except Exception as e:
        logger.error(f"Error in generate: {e}")


def generate_scores(
    model: Union[PreTrainedModel, 'LLM'],
    tokenizer: Optional[PreTrainedTokenizer],
    config: Dict[str, Any],
    iteration: int,
    responses_path: str,
    vllm: int = 0
) -> Union[str, None]:
    """
    为给定迭代生成评分并保存到输出路径。

    Args:
        model: 要使用的模型（可以是Transformers模型或vLLM模型）
        tokenizer: 分词器（仅用于Transformers模型）
        config: 配置字典
        iteration: 当前迭代次数
        responses_path: 回复数据的路径
        vllm: 是否使用vLLM（0：使用Transformers，1：使用vLLM）

    Returns:
        保存评分结果的输出路径，如果发生错误则返回None
    """
    try:
        logger.info(f"Generating scores for iteration {iteration}")
        output_dir = config["data_path"] / f"{iteration}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = output_dir / "gen_scores.jsonl"
        logger.info(f"Output path: {output_path}")

        gen_responses = pd.read_json(responses_path, lines=True)
        generate(
            model=model,
            tokenizer=tokenizer,
            gen_respones=gen_responses,
            output_path=output_path,
            vllm=vllm
        )
        return output_path
    except Exception as e:
        logger.error(f"Error in generate_scores: {e}")
        return None
