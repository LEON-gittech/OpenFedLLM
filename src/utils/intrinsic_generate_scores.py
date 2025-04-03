import os
import logging
import torch
import pandas as pd
from typing import List, Dict, Any, Union, Optional
from transformers import PreTrainedTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams

from src.utils.logging.logging_config import setup_logging

setup_logging()
logger = logging.getLogger()

template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{} 

### Response:"""

def compute_sequence_logprobs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    response: str,
) -> float:
    """
    计算给定响应的条件对数概率。使用累积的token级别对数概率。

    Args:
        model: 预训练模型
        tokenizer: 分词器
        prompt: 输入提示
        response: 模型响应

    Returns:
        响应序列的累积对数概率
    """
    try:
        # 构建输入
        full_prompt = template.format(prompt)
        
        # 对prompt和response分别编码
        prompt_ids = tokenizer(full_prompt, return_tensors="pt")["input_ids"].to(model.device)
        full_input = tokenizer(full_prompt + response, return_tensors="pt")
        full_input_ids = full_input["input_ids"].to(model.device)
        
        # 构建labels，prompt部分用-100屏蔽
        labels = torch.full_like(full_input_ids, -100)
        prompt_len = prompt_ids.shape[1]
        labels[:, prompt_len:] = full_input_ids[:, prompt_len:]
        
        # 获取模型输出
        # with torch.no_grad():
        outputs = model(full_input_ids)
        logits = outputs.logits[:, :-1, :]  # [batch, seq_len, vocab_size]
        labels = labels[:, 1:]  # 移除第一个token，因为我们shift了logits
        
        # 计算log softmax
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # 创建mask，只关注response部分
        labels_mask = (labels != -100)
        
        # 获取每个位置的实际token的log prob
        # 确保labels中的-100被替换为0，避免gather时的索引越界
        valid_labels = labels.clone()
        valid_labels[valid_labels == -100] = 0
        
        token_log_probs = torch.gather(
            log_probs, 
            dim=2, 
            index=valid_labels.unsqueeze(2)
        ).squeeze(2)
        
        # 只统计response部分的log prob总和
        sequence_log_prob = (token_log_probs * labels_mask).sum()
        
        return sequence_log_prob
        
    except Exception as e:
        logger.error(f"Error computing sequence logprobs: {e}")
        logger.error(f"Full error: {str(e)}")
        return float("-inf")

def generate(
    model: Union[PreTrainedModel, 'LLM'],
    tokenizer: Optional[PreTrainedTokenizer],
    gen_respones: pd.DataFrame,
    output_path: str,
    vllm: int = 0
) -> None:
    """
    基于模型的内在概率分布生成评分。

    Args:
        model: 当前模型
        ref_model: 参考模型
        tokenizer: 分词器
        gen_respones: 要评分的响应数据
        output_path: 保存评分结果的路径
        vllm: 是否使用vLLM
    """
    results = []
    intrinsic_rewards = []
    ref_intrinsic_rewards = []
    try:
        # if vllm:
        #     # TODO: 实现vLLM版本的概率计算
        #     logger.warning("vLLM version not implemented yet")
        #     return
        
        # else:
        # 对每个响应计算内在奖励
        # print(gen_respones)
        for _, row in gen_respones.iterrows():
            prompt = row["prompt"]
            prompt_id = row["prompt_id"]
            completion = row["completion"]

            # 计算当前模型和参考模型的对数概率
            current_logprob = compute_sequence_logprobs(
                model, tokenizer, prompt, completion
            )
            ref_logprob = compute_sequence_logprobs(
                ref_model, tokenizer, prompt, completion
            )
            
            # print(f"Reference: {ref_logprob:.3f}, Reward: {current_logprob:.3f}")
            # 计算内在奖励 r_θ(x,y) ∝ [log_π_θ(y|x) - log_π_ref(y|x)]

            intrinsic_rewards.append(current_logprob)
            ref_intrinsic_rewards.append(ref_logprob)
            results.append({
                "prompt_id": prompt_id,
                "prompt": prompt,
                "completion": completion,
                "score": (current_logprob-ref_logprob).item(),
            })

        # 保存结果
        df_results = pd.DataFrame(results)
        df_results.to_json(output_path, orient="records", lines=True)
        # print(intrinsic_rewards)
        # print(ref_intrinsic_rewards)
        return (torch.Tensor(intrinsic_rewards), torch.Tensor(ref_intrinsic_rewards))
        
    except Exception as e:
        logger.error(f"Error in generate: {e}")

def generate_scores(
    model: Union[PreTrainedModel, 'LLM'],
    ref_model: Union[PreTrainedModel, 'LLM'],
    tokenizer: Optional[PreTrainedTokenizer],
    config: Dict[str, Any],
    iteration: int,
    responses_path: str,
    vllm: int = 0
) -> Union[str, None]:
    """
    为给定迭代生成内在评分并保存到输出路径。

    Args:
        model: 当前模型
        ref_model: 参考模型
        tokenizer: 分词器
        config: 配置��典
        iteration: 当前迭代次数
        responses_path: 响应数据的路径
        vllm: 是否使用vLLM

    Returns:
        保存评分结果的输出路径，如果发生错误则返回None
    """
    try:
        logger.info(f"Generating intrinsic scores for iteration {iteration}")
        output_dir = config["data_path"] / f"{iteration}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = output_dir / "gen_scores.jsonl"
        logger.info(f"Output path: {output_path}")

        gen_responses = pd.read_json(responses_path, lines=True)
        generate(
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            gen_respones=gen_responses,
            output_path=output_path,
            vllm=vllm
        )
        return output_path
    except Exception as e:
        logger.error(f"Error in generate_scores: {e}")
        return None
