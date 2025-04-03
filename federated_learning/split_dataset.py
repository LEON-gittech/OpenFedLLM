import random
from datasets import Dataset

def split_dataset(fed_args, script_args, dataset):
    dataset = dataset.shuffle(seed=script_args.seed)        # Shuffle the dataset
    local_datasets = []
    if fed_args.split_strategy == "iid":
        for i in range(fed_args.num_clients):
            local_datasets.append(dataset.shard(fed_args.num_clients, i))
    
    return local_datasets

def get_dataset_this_round(dataset, round, fed_args, script_args):
    num2sample = script_args.batch_size * script_args.gradient_accumulation_steps * script_args.max_steps
    # print(num2sample)
    num2sample = min(num2sample, len(dataset))
    random.seed(round)
    random_idx = random.sample(range(0, len(dataset)), num2sample)
    dataset_this_round = dataset.select(random_idx)
    return dataset_this_round

def get_dataset_this_round_QA(dataset, round, fed_args, script_args):
    num2sample = script_args.batch_size * script_args.gradient_accumulation_steps * script_args.max_steps
    # print(num2sample)
    # 确保每轮都有base data
    num2sample = min(num2sample, len(dataset))-100
    random.seed(round)
    random_idx = list(range(100)) + random.sample(range(100, len(dataset)), num2sample)
    dataset_this_round = dataset.select(random_idx)
    dataset_this_round = dataset_this_round.shuffle(seed=round)
    return dataset_this_round

def get_dataset_this_round_fewshot(dataset: Dataset, round, fed_args, script_args):
    dataset_this_round = dataset.shard(fed_args.num_rounds, round)
    return dataset_this_round