import torch

class Config:
    @staticmethod
    def get_default_config(args):
        config = Config(args)
        config.NUM_EPOCHS = 2
        config.SAVE_LIMIT = 1
        config.BATCH_SIZE = 16
        config.SAVE_STEPS = 5_000
        config.EVAL_STEPS = 5_000
        config.LEARNING_RATE = 2e-5
        config.PER_DEVICE_BATCH_SIZE = 8
        config.FP_16 = torch.cuda.is_available()
        return config

    def __init__(self, args):
        self.MODEL_NAME = args.model
        self.NUM_EPOCHS = 0
        self.CONTEXT_LENGTH = 512
        self.SAVE_LIMIT = 0
        self.BATCH_SIZE = 0
        self.CHECKPOINT_PATH = f"{args.model}/" if args.checkpoint_path is None else args.checkpoint_path
        self.SAVE_STEPS = 0
        self.EVAL_STEPS = 0
        self.LEARNING_RATE = 0
        self.FP_16 = True
        self.MLM = 'roberta' in args.model
        self.PER_DEVICE_BATCH_SIZE = 0
        self.DATA_PATH = "data/"
        self.DATA_LOAD_SCRIPT = "dataset_load.py"
