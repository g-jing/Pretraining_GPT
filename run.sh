export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=2

python -m torch.distributed.launch\
    --nproc_per_node=8 main.py\
    --fp16 --model_size="small" \
    --loss_type="all" --batch_size=7 2>&1 | tee log.txt