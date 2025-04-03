import torch
import numpy as np
from typing import List, Tuple, Dict, Any
import json
import logging

logger = logging.getLogger(__name__)

def scores_to_ranks(scores: List[List[float]]) -> torch.Tensor:
    """
    将分数转换为排名。相同分数时，位置靠前的排名更高。
    
    Args:
        scores: 二维列表，每个子列表包含一组分数
        
    Returns:
        包含排名的tensor
    """
    ranks = []
    for score_group in scores:
        # 创建(分数, 位置)对的列表
        score_pos_pairs = [(score, idx) for idx, score in enumerate(score_group)]
        # 首先按分数降序排序，分数相同时按位置升序排序
        sorted_pairs = sorted(score_pos_pairs, key=lambda x: (-x[0], x[1]))
        # 创建排名映射
        rank_map = {pos: rank + 1 for rank, (_, pos) in enumerate(sorted_pairs)}
        # 按原始顺序获取排名
        group_ranks = [rank_map[i] for i in range(len(score_group))]
        ranks.append(group_ranks)
    
    return torch.tensor(ranks, dtype=torch.float32)

def ranknet_loss(scores1: List[List[float]], scores2: List[List[float]]) -> torch.Tensor:
    """
    计算RankNet Loss。
    
    Args:
        scores1: 第一个模型的分数
        scores2: 第二个模型的分数
        
    Returns:
        RankNet Loss值
    """
    # 将分数转换为排名
    ranks1 = scores_to_ranks(scores1)
    ranks2 = scores_to_ranks(scores2)
    
    batch_size, num_responses = ranks1.shape
    
    # 计算所有可能的响应对之间的排名差异
    total_loss = torch.tensor(0.0, requires_grad=True)
    
    for i in range(batch_size):
        for j in range(num_responses):
            for k in range(j + 1, num_responses):
                # 计算排名差异
                rank_diff1 = ranks1[i][j] - ranks1[i][k]
                rank_diff2 = ranks2[i][j] - ranks2[i][k]
                
                # 如果两个模型对排序有不同的判断
                if torch.sign(rank_diff1) != torch.sign(rank_diff2):
                    # 使用 sigmoid 函数计算概率
                    prob = 1 / (1 + torch.exp(-rank_diff1))
                    # 计算交叉熵损失
                    loss = -torch.log(prob) if rank_diff2 < 0 else -torch.log(1 - prob)
                    total_loss = total_loss + loss

    # 归一化损失
    num_pairs = batch_size * (num_responses * (num_responses - 1)) / 2
    return total_loss / num_pairs

import torch
import torch.nn.functional as F
from torch.linalg import svd

def preference_alignment_loss(local_scores, global_scores, lambda_s=0.2, tau=0.5):
    """
    实现包含符号敏感的逐样本评分对齐损失
    
    Args:
        local_scores: 本地模型评分 [batch_size, num_responses]
        global_scores: 全局模型评分 [batch_size, num_responses]
        lambda_s: 符号敏感系数 (默认0.2)
        tau: 温度系数 (默认0.5)
    """
    # 应用温度缩放
    local = local_scores / tau
    global_ = global_scores.detach() / tau
    
    # 绝对对齐项
    abs_loss = F.mse_loss(local, global_)
    
    # 符号敏感项
    sign_diff = torch.sign(local) != torch.sign(global_)
    sign_penalty = lambda_s * torch.abs(local**2 - global_**2) * sign_diff.float()
    
    return abs_loss + sign_penalty.mean()

def distribution_matching_loss(local_scores, global_scores, rank=2):
    """
    修正后的分布矩匹配损失函数，支持批量处理
    """
    # 均值对齐
    mu_loss = F.mse_loss(local_scores.mean(dim=1), global_scores.mean(dim=1))
    
    # 方差对齐
    var_loss = F.mse_loss(local_scores.var(dim=1), global_scores.var(dim=1))
    
    # 批量协方差计算
    def batch_lowrank_cov(x, y):
        # 添加特征维度 [B, N] -> [B, 1, N]
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        
        # 计算协方差矩阵 [B, N, N]
        cov_x = torch.matmul(x.transpose(1,2), x) / (x.size(2)-1)
        cov_y = torch.matmul(y.transpose(1,2), y) / (y.size(2)-1)
        
        # 批量SVD分解
        U_x, S_x, V_x = torch.linalg.svd(cov_x)
        U_y, S_y, V_y = torch.linalg.svd(cov_y)
        
        # 低秩近似 [B, N, rank]
        approx_x = U_x[:, :, :rank] @ torch.diag_embed(S_x[:, :rank]) @ V_x[:, :rank, :]
        approx_y = U_y[:, :, :rank] @ torch.diag_embed(S_y[:, :rank]) @ V_y[:, :rank, :]
        
        return F.mse_loss(approx_x, approx_y.detach())
    
    cov_loss = batch_lowrank_cov(local_scores, global_scores)
    
    return mu_loss + var_loss + cov_loss

def spa_regularization(local_scores, global_scores, lambda_pa=1.0, lambda_dm=0.5, **kwargs):
    """
    复合评分分布对齐正则项
    
    Args:
        local_scores: 本地模型评分 [B, N]
        global_scores: 全局模型评分 [B, N]
        lambda_pa: 逐样本对齐权重 (默认1.0)
        lambda_dm: 分布匹配权重 (默认0.5)
    """
    L_pa = preference_alignment_loss(local_scores, global_scores, **kwargs)
    L_dm = distribution_matching_loss(local_scores, global_scores, **kwargs)
    return lambda_pa * L_pa + lambda_dm * L_dm

def extract_scores_from_path(scores_path: str) -> Dict[str, List[float]]:
    """
    从文件中提取分数。
    
    Args:
        scores_path: 分数文件的路径
        
    Returns:
        按prompt_id分组的分数列表字典
    """
    prompts: Dict[str, List[Dict[str, Any]]] = {}
    
    try:
        with open(scores_path, "r") as f:
            for line in f:
                data = json.loads(line)
                prompt_id = data["prompt_id"]
                if prompt_id not in prompts:
                    prompts[prompt_id] = []
                prompts[prompt_id].append(data)
        
        # 将分组后的数据转换为分数列表
        scores_dict = {}
        for prompt_id, prompt_group in prompts.items():
            # 按原始顺序排列分数
            scores = [p["score"] for p in sorted(prompt_group, key=lambda x: x["completion"])]
            scores_dict[prompt_id] = scores
            
        return scores_dict
    except Exception as e:
        logger.error(f"Error extracting scores from path: {e}")
        return {}

from itertools import product
import torch
from torch.nn import BCEWithLogitsLoss
def rankNet(y_pred, y_true, padded_value_indicator=-1, weight_by_diff=False, weight_by_diff_powed=False):
    """
    RankNet loss introduced in "Learning to Rank using Gradient Descent".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param weight_by_diff: flag indicating whether to weight the score differences by ground truth differences.
    :param weight_by_diff_powed: flag indicating whether to weight the score differences by the squared ground truth differences.
    :return: loss value, a torch.Tensor
    """
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    mask = y_true == padded_value_indicator
    y_pred[mask] = float('-inf')
    y_true[mask] = float('-inf')

    # here we generate every pair of indices from the range of document length in the batch
    document_pairs_candidates = list(product(range(y_true.shape[1]), repeat=2))

    pairs_true = y_true[:, document_pairs_candidates]
    selected_pred = y_pred[:, document_pairs_candidates]

    # here we calculate the relative true relevance of every candidate pair
    true_diffs = pairs_true[:, :, 0] - pairs_true[:, :, 1]
    pred_diffs = selected_pred[:, :, 0] - selected_pred[:, :, 1]

    # here we filter just the pairs that are 'positive' and did not involve a padded instance
    # we can do that since in the candidate pairs we had symetric pairs so we can stick with
    # positive ones for a simpler loss function formulation
    the_mask = (true_diffs > 0) & (~torch.isinf(true_diffs))

    pred_diffs = pred_diffs[the_mask]

    weight = None
    if weight_by_diff:
        abs_diff = torch.abs(true_diffs)
        weight = abs_diff[the_mask]
    elif weight_by_diff_powed:
        true_pow_diffs = torch.pow(pairs_true[:, :, 0], 2) - torch.pow(pairs_true[:, :, 1], 2)
        abs_diff = torch.abs(true_pow_diffs)
        weight = abs_diff[the_mask]

    # here we 'binarize' true relevancy diffs since for a pairwise loss we just need to know
    # whether one document is better than the other and not about the actual difference in
    # their relevancy levels
    true_diffs = (true_diffs > 0).type(torch.float32)
    true_diffs = true_diffs[the_mask]

    return BCEWithLogitsLoss(weight=weight)(pred_diffs, true_diffs)

def compute_ranknet_loss(scores_path1: str, scores_path2: str) -> torch.Tensor:
    """
    从两个路径读取分数并计算RankNet Loss。
    
    Args:
        scores_path1: 第一个模型的分数文件路径
        scores_path2: 第二个模型的分数文件路径
        
    Returns:
        计算得到的RankNet Loss
    """
    try:
        # 从文件中提取分数
        scores_dict1 = extract_scores_from_path(scores_path1)
        scores_dict2 = extract_scores_from_path(scores_path2)
        
        # 确保两个文件包含相同的prompt_ids
        common_prompt_ids = set(scores_dict1.keys()) & set(scores_dict2.keys())
        
        if not common_prompt_ids:
            raise ValueError("No common prompt_ids found between the two score files")
        
        # 将分数转换为列表格式
        scores1 = [scores_dict1[pid] for pid in common_prompt_ids]
        scores2 = [scores_dict2[pid] for pid in common_prompt_ids]
        
        # 计算RankNet Loss
        # return ranknet_loss(scores1, scores2)
        return rankNet(torch.tensor(scores1, requires_grad=True, dtype=torch.float), torch.tensor(scores2, requires_grad=True, dtype=torch.float))
    except Exception as e:
        logger.error(f"Error computing ranknet loss: {e}")
        return torch.tensor(0.0)

def compute_spa_loss(scores_path1: str, scores_path2: str) -> torch.Tensor:
    """
    从两个路径读取分数并计算RankNet Loss。
    
    Args:
        scores_path1: 第一个模型的分数文件路径
        scores_path2: 第二个模型的分数文件路径
        
    Returns:
        计算得到的RankNet Loss
    """
    try:
        # 从文件中提取分数
        scores_dict1 = extract_scores_from_path(scores_path1)
        scores_dict2 = extract_scores_from_path(scores_path2)
        
        # 确保两个文件包含相同的prompt_ids
        common_prompt_ids = set(scores_dict1.keys()) & set(scores_dict2.keys())
        
        if not common_prompt_ids:
            raise ValueError("No common prompt_ids found between the two score files")
        
        # 将分数转换为列表格式
        scores1 = torch.tensor([scores_dict1[pid] for pid in common_prompt_ids], dtype=torch.float, requires_grad=True)
        scores2 = torch.tensor([scores_dict2[pid] for pid in common_prompt_ids], dtype=torch.float, requires_grad=True)
        
        # 计算RankNet Loss
        return spa_regularization(scores1, scores2)
    except Exception as e:
        logger.error(f"Error computing ranknet loss: {e}")
        return torch.tensor(0.0)

# 测试代码
if __name__ == "__main__":
    # 测试用例
    scores1 = [
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [1, 2, 3, 4]
    ]
    
    scores2 = [
        [4,3,2,1],
        [4,3,2,1],
        [4,3,2,1],
        [4,3,2,1],
        [4,3,2,1]
    ]
    
    # 测试排名转换
    ranks1 = scores_to_ranks(scores1)
    print("Ranks for scores1:")
    print(ranks1)
    
    ranks2 = scores_to_ranks(scores2)
    print("\nRanks for scores2:")
    print(ranks2)
    
    # 测试损失计算
    loss = ranknet_loss(scores1, scores2)
    print("\nRankNet Loss:", loss.item())
