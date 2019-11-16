class Config():
# if value is None, the value will be the default value in args.  

    train_data_file = None
    output_dir = None
    eval_data_file = None
    model_type = "gpt2"
    model_name_or_path = 'gpt2' # if gpu memory enough, choose a bigger one
    #mlm = None
    #mlm_probability = None
    config_name = None # optional when model_name_or_path exist
    tokenizer_name = None # optional when model_name_or_path exist
    cache_dir = None # Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)
    block_size = None # Optional input sequence length after tokenization. "The training dataset will be truncated.
    
    # true or false
    do_train = None
    do_eval = None
    evaluate_during_training = None
    do_lower_case = None
    
    per_gpu_train_batch_size = None
    per_gpu_eval_batch_size = None
    gradient_accumulation_steps = None
    learning_rate = None
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
    no_cuda = None
    overwrite_output_dir = None
    overwrite_cache = None

    seed = None

    # true or false
    fp16 = None

    fp16_opt_level = None
    local_rank = None
    server_ip = None
    server_port = None

