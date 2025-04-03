import pandas as pd
import re
import uuid
import logging
from transformers import TextStreamer
import os
from typing import List, Dict, Any, Union, Optional
from transformers import PreTrainedTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams  # 添加vLLM相关导入
from rouge_score import rouge_scorer

from src.utils.read_write_jsonl import read_jsonl_file, write_jsonl_file
from src.utils.prompts import prompt_step_01
from src.utils.logging.logging_config import setup_logging
import random

setup_logging()
logger = logging.getLogger()


def get_random_prompts(data: pd.DataFrame, num_prompts: int = 4, prev_prompts = []) -> List[str]:
    """
    Gets a random sample of prompts from the data.

    Args:
        data: The data to get the prompts from.
        num_prompts: The number of prompts to get.

    Returns:
        A list of random prompts.
    """
    try:
        if prev_prompts == []:
            return data.sample(n=num_prompts)["prompt"].tolist()
        else:
            random_prompts = random.sample(prev_prompts, 2) + data.sample(n=num_prompts-2)["prompt"].tolist()
            random.shuffle(random_prompts)
            return random_prompts
    except Exception as e:
        print(prev_prompts)
        print(data.sample(n=num_prompts-2)["prompt"].tolist())
        logger.error(f"Error getting random prompts: {e}")
        return []


def generate_prompt(examples: List[str]) -> str:
    """
    Generates a prompt from the examples.

    Args:
        examples: The examples to generate the prompt from.

    Returns:
        The generated prompt.
    """
    try:
        # 创建新的提示字符串，而不是修改全局变量
        prompt = prompt_step_01
        for item in examples:
            prompt += f"<task>|{item}</task>\n"
        return prompt
    except Exception as e:
        logger.error(f"Error generating prompt: {e}")
        return prompt_step_01


def extract_prompt(answer: str) -> List[str]:
    """
    Extracts the prompts from the answer.

    Args:
        answer: The answer to extract the prompts from.

    Returns:
        A list of extracted prompts.
    """
    prompts = []
    try:
        extracted_prompts = re.findall(r"<task>\|(.*?)</task>", answer, re.DOTALL)
        for prompt in extracted_prompts:
            prompts.append(prompt)
    except Exception as e:
        logger.error(f"Error extracting prompts: {e}")
    return prompts


def do_sample(
    model, tokenizer: PreTrainedTokenizer, task_prompts: List[str]
) -> str:
    """
    Samples from the model using the task prompts.

    Args:
        model: The model to sample from.
        tokenizer: The tokenizer to use.
        task_prompts: The task prompts to use.

    Returns:
        The sampled text.
    """
    try:
        prompt = generate_prompt(task_prompts)
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
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return decoded[0]
    except Exception as e:
        print(f"model input prompt {prompt}")
        logger.error(f"Error during sampling: {e}")
        return ""


def check_rouge_similarity(new_prompt: str, prev_prompts: List[str], threshold: float = 0.7) -> bool:
    """
    检查新prompt与已有prompts的ROUGE-L相似度。

    Args:
        new_prompt: 新生成的prompt
        prev_prompts: 已有的prompts列表
        threshold: 相似度阈值

    Returns:
        如果与所有已有prompts的相似度都小于阈值，返回True；否则返回False
    """
    try:
        if not prev_prompts:
            return True
            
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        for prev_prompt in prev_prompts:
            scores = scorer.score(new_prompt, prev_prompt)
            if scores['rougeL'].fmeasure >= threshold:
                return False
        return True
    except Exception as e:
        logger.error(f"Error checking ROUGE similarity: {e}")
        return False


def generate(
    model: Union[PreTrainedModel, 'LLM'],
    tokenizer: Optional[PreTrainedTokenizer],
    ift_data: pd.DataFrame,
    new_prompts_to_generate: int,
    vllm: int = 0,
    prev_prompts: List[str] = None,
    few_shot_num: int = 8,
) -> List[Dict[str, Any]]:
    """
    生成新的提示。

    Args:
        model: 要使用的模型（可以是Transformers模型或vLLM模型）
        tokenizer: 分词器（仅用于Transformers模型）
        ift_data: IFT数据
        new_prompts_to_generate: 要生成的新提示数量
        vllm: 是否使用vLLM
        prev_prompts: 之前生成的prompts列表

    Returns:
        新生成的提示列表
    """
    uniq_prompts = set()
    new_prompts = []
    prev_prompts = prev_prompts or []
    
    try:
        if vllm:
            # task_prompts_list = [get_random_prompts(ift_data) for _ in range(10*new_prompts_to_generate)]
            # prompts_list = [generate_prompt(task_prompts) for task_prompts in task_prompts_list]
            
            # sampling_params = SamplingParams(
            #     temperature=0.6,
            #     top_p=0.9,
            #     max_tokens=256
            # )

            # if hasattr(model, "fast_generate"):
            #     outputs = model.fast_generate(
            #         prompts_list,
            #         sampling_params = sampling_params,
            #         lora_request = model.load_lora("/mnt/bn/merlin-datavolume-tsy/leon/FedSPA/checkpoints_cache/saved_model_lora"),
            #     )
            # else:
            #     outputs = model.generate(prompts_list, sampling_params)
            
            # for output in outputs:
            #     prompts = extract_prompt(output.outputs[0].text)
            #     for prompt in prompts:
            #         # 检查是否已存在且ROUGE-L相似度是否过高
            #         if prompt not in uniq_prompts and check_rouge_similarity(prompt, prev_prompts):
            #             uniq_prompts.add(prompt)
            #             prompt_id = str(uuid.uuid4())
            #             new_prompts.append(
            #                 {"id": prompt_id, "prompt": prompt, "source": "generated"}
            #             )
            #             if len(uniq_prompts) >= new_prompts_to_generate:
            #                 break
            #     if len(uniq_prompts) >= new_prompts_to_generate:
            #         break
            while len(uniq_prompts) < new_prompts_to_generate:
                # 生成新的 prompts
                task_prompts_list = [get_random_prompts(ift_data) for _ in range(new_prompts_to_generate)]
                prompts_list = [generate_prompt(task_prompts) for task_prompts in task_prompts_list]
                
                sampling_params = SamplingParams(
                    temperature=0.6,
                    top_p=0.9,
                    max_tokens=256
                )

                if hasattr(model, "fast_generate"):
                    outputs = model.fast_generate(
                        prompts_list,
                        sampling_params=sampling_params,
                        lora_request=model.load_lora("/mnt/bn/merlin-datavolume-tsy/leon/FedSPA/checkpoints_cache/saved_model_lora"),
                    )
                else:
                    outputs = model.generate(prompts_list, sampling_params)
                
                for output in outputs:
                    prompts = extract_prompt(output.outputs[0].text)
                    for prompt in prompts:
                        # 检查是否已存在且ROUGE-L相似度是否过高
                        if prompt not in uniq_prompts and check_rouge_similarity(prompt, prev_prompts):
                            uniq_prompts.add(prompt)
                            prompt_id = str(uuid.uuid4())
                            new_prompts.append(
                                {"id": prompt_id, "prompt": prompt, "source": "generated"}
                            )
                            if len(uniq_prompts) >= new_prompts_to_generate:
                                break
                    if len(uniq_prompts) >= new_prompts_to_generate:
                        break
        else:
            while len(uniq_prompts) < new_prompts_to_generate:
                task_prompts = get_random_prompts(ift_data, num_prompts=few_shot_num, prev_prompts=prev_prompts)
                assert len(task_prompts) == few_shot_num
                answer = do_sample(model, tokenizer, task_prompts)
                if answer == "": return []
                prompts = extract_prompt(answer)
                for prompt in prompts:
                    # 检查是否已存在且ROUGE-L相似度是否过高
                    if prompt not in uniq_prompts and check_rouge_similarity(prompt, prev_prompts):
                        uniq_prompts.add(prompt)
                        prompt_id = str(uuid.uuid4())
                        new_prompts.append(
                            {"id": prompt_id, "prompt": prompt, "source": "generated"}
                        )
                        if len(uniq_prompts) >= new_prompts_to_generate:
                            break
            print(f"prompts {uniq_prompts}\n\n")
        
        return new_prompts
    except Exception as e:
        logger.error(f"Error generating new prompts: {e}")
        return []


def generate_new_prompts(
    model: Union[PreTrainedModel, 'LLM'],
    tokenizer: Optional[PreTrainedTokenizer],
    config: Dict[str, Any],
    iteration: int,
    vllm: int = 0
) -> Union[str, None]:
    """
    为给定迭代生成新提示并保存到输出路径。

    Args:
        model: 要使用的模型（可以是Transformers模型或vLLM模型）
        tokenizer: 分词器（仅用于Transformers模型）
        config: 配置字典
        iteration: 当前迭代次数
        vllm: 是否使用vLLM（0：使用Transformers，1：使用vLLM）

    Returns:
        保存新提示的输出路径，如果发生错误则返回None
    """
    try:
        logger.info(f"Generating new prompts for iteration {iteration}")
        ift_data = read_jsonl_file(config["ift_data_path"] / config["ift_dataset"])
        new_prompts = generate(
            model=model,
            tokenizer=tokenizer,
            ift_data=ift_data,
            new_prompts_to_generate=config["generate_prompts"]["new_prompts"],
            vllm=vllm
        )
        logger.info(f"Generated {len(new_prompts)} new prompts")
        new_prompts_df = pd.DataFrame(new_prompts)
        output_dir = config["data_path"] / f"{iteration}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = output_dir / "gen_prompts.jsonl"
        write_jsonl_file(new_prompts_df, output_path)
        return output_path
    except Exception as e:
        logger.error(f"Error in generate_new_prompts: {e}")
        return None