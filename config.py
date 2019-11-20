class Config():
# if value is None, the value will be the default value in args.  
    def __init__(self):

        self.train_batch_size = 8

        self.total_datum = 140000000
        self.step_per_batch = 10000
        self.gpu_count = 8

        self.train_data_file = "train.jsonl" # training dataset
        self.output_dir = "output" # The output directory where the model predictions and checkpoints will be written.
        self.eval_data_file = "eval.jsonl" # eval dataset

        # cache_dir = None # Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)
        self.block_size = 1024 # Optional input sequence length after tokenization. "The training dataset will be truncated.
        
        # true or false
        self.do_train = True # True
        self.do_eval = True # True
        self.evaluate_during_training = True # True
        self.do_lower_case = False # False
        
        self.per_gpu_train_batch_size = 8 # or 16 # depends on total step
        self.per_gpu_eval_batch_size = 8 # at large as possible # depends on gpu memory
        self.gradient_accumulation_steps = 1 # depend on total step
        self.learning_rate = 1e-5
        self.weight_decay = 0.01
        self.adam_epsilon = 1e-6 
        self.max_grad_norm = 1.0
        self.num_train_epochs = 10 # but stop at two
        self.max_steps = 10000 
        self.warmup_steps = 100 # 1 epoch
        self.logging_steps = 50
        self.save_steps = 50 # save model during training
        self.save_total_limit = None # how many model to keeps
        
        self.mask_gradient_normalization = 1.0

        # true or false
        self.eval_all_checkpoints = True # eval all model 
        self.no_cuda = False # False
        self.overwrite_output_dir = True
        self.overwrite_cache = True

        self.seed = 42
        
        self.fp16 = True # spped up but lose precision

        self.fp16_opt_level = "O1"
        self.local_rank = 1
        self.server_ip = ""
        self.server_port = ""

    def hp_calculate(self):

        self.batch_size = self.total_datum // self.step_per_batch
        
        self.gradient_accumulation_steps = self.batch_size // self.gpu_count

        self.stop_batch = 2

        self.stop_step = 20000
        

        



