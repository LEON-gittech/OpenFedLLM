#——————————————————————————————————————————————- 10 clients ————————————————————————————————————————————————
CUDA_VISIBLE_DEVICES=0 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 10 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_code_0.01 --seed 42 > 1.out &

CUDA_VISIBLE_DEVICES=2 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 10 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_code_0.1 --seed 42 > 2.out &

CUDA_VISIBLE_DEVICES=3 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 10 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_code_1 --seed 42 > 3.out &

wait

CUDA_VISIBLE_DEVICES=0 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 10 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_code_10 --seed 42 > 4.out &


CUDA_VISIBLE_DEVICES=3 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 10 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_code_0.01 --seed 43 > 5.out &

CUDA_VISIBLE_DEVICES=2 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 10 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_code_0.1 --seed 43 > 6.out &

wait


CUDA_VISIBLE_DEVICES=3 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 10 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_code_1 --seed 43 > 7.out &

CUDA_VISIBLE_DEVICES=0 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 10 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_code_10 --seed 43 > 8.out &


CUDA_VISIBLE_DEVICES=2 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 10 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_code_0.01 --seed 44 > 9.out &

wait


CUDA_VISIBLE_DEVICES=2 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 10 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_code_0.1 --seed 44 > 10.out &

CUDA_VISIBLE_DEVICES=3 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 10 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_code_1 --seed 44 > 11.out &

CUDA_VISIBLE_DEVICES=0 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 10 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_code_10 --seed 44 > 12.out &

wait

CUDA_VISIBLE_DEVICES=0 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 10 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_med_0.01 --seed 42 > 1.out &

CUDA_VISIBLE_DEVICES=2 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 10 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_med_0.1 --seed 42 > 2.out &

CUDA_VISIBLE_DEVICES=3 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 10 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_med_1 --seed 42 > 3.out &

wait

CUDA_VISIBLE_DEVICES=0 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 10 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_med_10 --seed 42 > 4.out &


CUDA_VISIBLE_DEVICES=3 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 10 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_med_0.01 --seed 43 > 5.out &

CUDA_VISIBLE_DEVICES=2 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 10 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_med_0.1 --seed 43 > 6.out &

wait

CUDA_VISIBLE_DEVICES=3 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 10 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_med_1 --seed 43 > 7.out &

CUDA_VISIBLE_DEVICES=2 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 10 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_med_10 --seed 43 > 8.out &


CUDA_VISIBLE_DEVICES=0 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 10 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_med_0.01 --seed 44 > 9.out &

wait

CUDA_VISIBLE_DEVICES=2 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 10 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_med_0.1 --seed 44 > 10.out &

CUDA_VISIBLE_DEVICES=3 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 10 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_med_1 --seed 44 > 11.out &

CUDA_VISIBLE_DEVICES=0 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 10 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_med_10 --seed 44 > 12.out &

wait

CUDA_VISIBLE_DEVICES=0 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 10 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_fin_0.01 --seed 42 > 1.out &

CUDA_VISIBLE_DEVICES=2 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 10 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_fin_0.1 --seed 42 > 2.out &

CUDA_VISIBLE_DEVICES=3 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 10 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_fin_1 --seed 42 > 3.out &

wait

CUDA_VISIBLE_DEVICES=0 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 10 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_fin_10 --seed 42 > 4.out &


CUDA_VISIBLE_DEVICES=3 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 10 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_fin_0.01 --seed 43 > 5.out &

CUDA_VISIBLE_DEVICES=2 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 10 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_fin_0.1 --seed 43 > 6.out &

wait

CUDA_VISIBLE_DEVICES=3 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 10 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_fin_1 --seed 43 > 7.out &

CUDA_VISIBLE_DEVICES=2 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 10 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_fin_10 --seed 43 > 8.out &


CUDA_VISIBLE_DEVICES=0 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 10 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_fin_0.01 --seed 44 > 9.out &

wait

CUDA_VISIBLE_DEVICES=2 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 10 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_fin_0.1 --seed 44 > 10.out &

CUDA_VISIBLE_DEVICES=3 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 10 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_fin_1 --seed 44 > 11.out &

CUDA_VISIBLE_DEVICES=0 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 10 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_fin_10 --seed 44 > 12.out &

wait

CUDA_VISIBLE_DEVICES=0 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 10 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_math_0.01 --seed 42 > 1.out &

CUDA_VISIBLE_DEVICES=2 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 10 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_math_0.1 --seed 42 > 2.out &

CUDA_VISIBLE_DEVICES=3 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 10 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_math_1 --seed 42 > 3.out &

wait

CUDA_VISIBLE_DEVICES=0 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 10 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_math_10 --seed 42 > 4.out &


CUDA_VISIBLE_DEVICES=3 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 10 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_math_0.01 --seed 43 > 5.out &

CUDA_VISIBLE_DEVICES=2 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 10 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_math_0.1 --seed 43 > 6.out &

wait

CUDA_VISIBLE_DEVICES=3 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 10 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_math_1 --seed 43 > 7.out &

CUDA_VISIBLE_DEVICES=2 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 10 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_math_10 --seed 43 > 8.out &


CUDA_VISIBLE_DEVICES=0 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 10 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_math_0.01 --seed 44 > 9.out &

wait

CUDA_VISIBLE_DEVICES=2 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 10 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_math_0.1 --seed 44 > 10.out &

CUDA_VISIBLE_DEVICES=3 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 10 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_math_1 --seed 44 > 11.out &

CUDA_VISIBLE_DEVICES=0 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 10 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_math_10 --seed 44 > 12.out &
wait

#——————————————————————————————————————————————- 10 clients ————————————————————————————————————————————————



#——————————————————————————————————————————————- 100 clients ————————————————————————————————————————————————
CUDA_VISIBLE_DEVICES=0 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 100 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_code_0.01_100 --seed 42 > 1.out &

CUDA_VISIBLE_DEVICES=2 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 100 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_code_0.1_100 --seed 42 > 2.out &

CUDA_VISIBLE_DEVICES=3 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 100 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_code_1_100 --seed 42 > 3.out &

wait

CUDA_VISIBLE_DEVICES=0 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 100 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_code_10_100 --seed 42 > 4.out &


CUDA_VISIBLE_DEVICES=3 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 100 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_code_0.01_100 --seed 43 > 5.out &

CUDA_VISIBLE_DEVICES=2 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 100 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_code_0.1_100 --seed 43 > 6.out &

wait

CUDA_VISIBLE_DEVICES=3 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 100 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_code_1_100 --seed 43 > 7.out &

CUDA_VISIBLE_DEVICES=2 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 100 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_code_10_100 --seed 43 > 8.out &


CUDA_VISIBLE_DEVICES=0 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 100 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_code_0.01_100 --seed 44 > 9.out &

wait

CUDA_VISIBLE_DEVICES=2 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 100 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_code_0.1_100 --seed 44 > 10.out &

CUDA_VISIBLE_DEVICES=3 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 100 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_code_1_100 --seed 44 > 11.out &

CUDA_VISIBLE_DEVICES=0 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 100 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_code_10_100 --seed 44 > 12.out &

wait


CUDA_VISIBLE_DEVICES=0 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 100 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_med_0.01_100 --seed 42 > 1.out &

CUDA_VISIBLE_DEVICES=2 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 100 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_med_0.1_100 --seed 42 > 2.out &

CUDA_VISIBLE_DEVICES=3 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 100 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_med_1_100 --seed 42 > 3.out &

wait

CUDA_VISIBLE_DEVICES=0 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 100 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_med_10_100 --seed 42 > 4.out &


CUDA_VISIBLE_DEVICES=3 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 100 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_med_0.01_100 --seed 43 > 5.out &

CUDA_VISIBLE_DEVICES=2 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 100 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_med_0.1_100 --seed 43 > 6.out &

wait

CUDA_VISIBLE_DEVICES=3 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 100 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_med_1_100 --seed 43 > 7.out &

CUDA_VISIBLE_DEVICES=2 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 100 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_med_10_100 --seed 43 > 8.out &


CUDA_VISIBLE_DEVICES=0 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 100 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_med_0.01_100 --seed 44 > 9.out &

wait

CUDA_VISIBLE_DEVICES=2 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 100 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_med_0.1_100 --seed 44 > 10.out &

CUDA_VISIBLE_DEVICES=3 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 100 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_med_1_100 --seed 44 > 11.out &

CUDA_VISIBLE_DEVICES=0 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 100 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_med_10_100 --seed 44 > 12.out &

wait

CUDA_VISIBLE_DEVICES=0 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 100 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_fin_0.01_100 --seed 42 > 1.out &

CUDA_VISIBLE_DEVICES=2 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 100 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_fin_0.1_100 --seed 42 > 2.out &

CUDA_VISIBLE_DEVICES=3 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 100 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_fin_1_100 --seed 42 > 3.out &

wait

CUDA_VISIBLE_DEVICES=0 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 100 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_fin_10_100 --seed 42 > 4.out &


CUDA_VISIBLE_DEVICES=3 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 100 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_fin_0.01_100 --seed 43 > 5.out &

CUDA_VISIBLE_DEVICES=2 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 100 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_fin_0.1_100 --seed 43 > 6.out &

wait

CUDA_VISIBLE_DEVICES=3 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 100 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_fin_1_100 --seed 43 > 7.out &

CUDA_VISIBLE_DEVICES=2 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 100 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_fin_10_100 --seed 43 > 8.out &


CUDA_VISIBLE_DEVICES=0 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 100 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_fin_0.01_100 --seed 44 > 9.out &

wait

CUDA_VISIBLE_DEVICES=2 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 100 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_fin_0.1_100 --seed 44 > 10.out &

CUDA_VISIBLE_DEVICES=3 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 100 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_fin_1_100 --seed 44 > 11.out &

CUDA_VISIBLE_DEVICES=0 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 100 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_fin_10_100 --seed 44 > 12.out &

wait

CUDA_VISIBLE_DEVICES=0 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 100 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_math_0.01_100 --seed 42 > 1.out &

CUDA_VISIBLE_DEVICES=2 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 100 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_math_0.1_100 --seed 42 > 2.out &

CUDA_VISIBLE_DEVICES=3 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 100 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_math_1_100 --seed 42 > 3.out &

wait

CUDA_VISIBLE_DEVICES=0 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 100 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_math_10_100 --seed 42 > 4.out &


CUDA_VISIBLE_DEVICES=3 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 100 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_math_0.01_100 --seed 43 > 5.out &

CUDA_VISIBLE_DEVICES=2 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 100 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_math_0.1_100 --seed 43 > 6.out &

wait

CUDA_VISIBLE_DEVICES=3 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 100 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_math_1_100 --seed 43 > 7.out &

CUDA_VISIBLE_DEVICES=2 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 100 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_math_10_100 --seed 43 > 8.out &

CUDA_VISIBLE_DEVICES=0 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 100 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_math_0.01_100 --seed 44 > 9.out &

wait

CUDA_VISIBLE_DEVICES=2 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 100 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_math_0.1_100 --seed 44 > 10.out &

CUDA_VISIBLE_DEVICES=3 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 100 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_math_1_100 --seed 44 > 11.out &

CUDA_VISIBLE_DEVICES=0 nohup python3 main_sft.py --learning_rate 5e-5 --model_name_or_path /mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/ --dataset_sample 20000 --fed_alg fedavg --num_clients 100 --sample_clients 2 --max_steps 10 --num_rounds 30 --batch_size 8 --gradient_accumulation_steps 4 --seq_length 2048 --peft_lora_r 32 --peft_lora_alpha 64 --use_peft --load_in_4bit --output_dir /mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed --template alpaca --unsloth 1 --bf16 1 --seq_length 2048 --dataset_name niid_math_10_100 --seed 44 > 12.out &
#——————————————————————————————————————————————- 100 clients ————————————————————————————————————————————————