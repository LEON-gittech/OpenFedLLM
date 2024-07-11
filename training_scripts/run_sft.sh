max_steps=10
num_rounds=50
batch_size=16
gradient_accumulation_steps=1
seq_length=2048
num_clients=10
sample_clients=2
lora_r=16
lora_alpha=16   # twice of lora_r
lr=5e-5

# local_data_dir=""       # you may uncomment this line if your data is stored locally and include it in the python command
dataset_name="neg"
dataset_sample=20000
model_name_or_path="/mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/" # /mnt/bn/data-tns-live-llm/leon/datasets/Meta-Llama-3-8B/
output_dir="/mnt/bn/data-tns-live-llm/leon/datasets/fed" 

gpu=1
fed_alg="fedavg"

CUDA_VISIBLE_DEVICES=$gpu python3 main_sft.py \
 --learning_rate $lr \
 --model_name_or_path $model_name_or_path \
 --dataset_name $dataset_name \
 --dataset_sample $dataset_sample \
 --fed_alg $fed_alg \
 --num_clients $num_clients \
 --sample_clients $sample_clients \
 --max_steps $max_steps \
 --num_rounds $num_rounds \
 --batch_size $batch_size \
 --gradient_accumulation_steps $gradient_accumulation_steps \
 --seq_length $seq_length \
 --peft_lora_r $lora_r \
 --peft_lora_alpha $lora_alpha \
 --use_peft \
 --load_in_4bit \
 --output_dir $output_dir \
 --template "alpaca" \
 --unsloth 1 \
 --bf16 1 \
 --seq_length $seq_length