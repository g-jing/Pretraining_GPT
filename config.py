class Config():
# if value is None, the value will be the default value in args.  

    train_data_file = "train.jsonl"
    output_dir = "output"
    eval_data_file = "eval.jsonl"
    # model_type = "gpt2"
    # model_name_or_path = 'gpt2' # 'gpt2-medium', 'gpt2-large' if gpu memory enough, choose a bigger one
    # config_name = None # optional when model_name_or_path exist
    tokenizer_name = None # optional when model_name_or_path exist
    cache_dir = None # Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)
    block_size = None # Optional input sequence length after tokenization. "The training dataset will be truncated.
    
    # true or false
    do_train = True # True
    do_eval = True # True
    evaluate_during_training = True # True
    do_lower_case = False# False
    
    per_gpu_train_batch_size = None # depends on total step
    per_gpu_eval_batch_size = None # depends on gpu memory
    gradient_accumulation_steps = None # depend on total step
    learning_rate = None #TODO these hp, ask qingyang wu
    weight_decay = None
    adam_epsilon = None
    max_grad_norm = None
    num_train_epochs = None
    max_steps = None
    warmup_steps = None
    logging_steps = None
    save_steps = None
    save_total_limit = None

    # true or false
    eval_all_checkpoints = None
    no_cuda = False # False
    overwrite_output_dir = None
    overwrite_cache = None

    seed = 42

    # true or false
    fp16 = False # spped up but lose precision

    fp16_opt_level = None
    local_rank = None
    server_ip = None
    server_port = None

