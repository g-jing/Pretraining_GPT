class Config():
# if value is None, the value will be the default value in args.  

    train_data_file = "train.jsonl" # training dataset
    output_dir = "output" # The output directory where the model predictions and checkpoints will be written.
    eval_data_file = "eval.jsonl" # eval dataset

    # cache_dir = None # Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)
    block_size = 1024 # Optional input sequence length after tokenization. "The training dataset will be truncated.
    
    # true or false
    do_train = True # True
    do_eval = True # True
    evaluate_during_training = True # True
    do_lower_case = False # False
    
    per_gpu_train_batch_size = 8 # or 16 # depends on total step
    per_gpu_eval_batch_size = 8 # at large as possible # depends on gpu memory
    gradient_accumulation_steps = 1 # depend on total step
    learning_rate = 1e-5 #TODO these hp, ask qingyang wu
    weight_decay = 0.01
    adam_epsilon = 1e-6 
    max_grad_norm = 1.0
    num_train_epochs = 10 # but stop at two
    max_steps = 10000 
    warmup_steps = 100 # 1 epoch
    logging_steps = 50
    save_steps = 50 # save model during training
    save_total_limit = None # how many model to keeps
    
    mask_gradient_normalization = 1.0

    # true or false
    eval_all_checkpoints = True # eval all model 
    no_cuda = False # False
    overwrite_output_dir = True
    overwrite_cache = True

    seed = 42
    
    fp16 = True # spped up but lose precision

    fp16_opt_level = "O1"
    local_rank = -1
    server_ip = ""
    server_port = ""

