CUDA_VISIBLE_DEVICES=2,3,4,5 python -m torch.distributed.launch --nproc_per_node=4 main_manager.py --fp16 --train_data_file="small_reddit.jsonl"
