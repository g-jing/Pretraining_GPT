export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2

python -m torch.distributed.launch\
    --nproc_per_node=1 main.py\
    --fp16 --model_size="small" \
    --loss_type="all" --batch_size=11 \
    --kl_model_size="small" 2>&1 | tee log.txt