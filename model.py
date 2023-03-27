import os
import json
import glob
import torch
from transformers import GPT2LMHeadModel, AutoConfig
from transformers import RobertaConfig, RobertaForMaskedLM
from datasets import load_dataset
from transformers import AutoTokenizer, RobertaTokenizerFast
from tokenizers import ByteLevelBPETokenizer
from transformers import DataCollatorForLanguageModeling
from config import Config
from transformers import Trainer, TrainingArguments

import numpy as np


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


class Model:
    def __init__(self, config: Config):
        self.config = config
        self._init_dataset()

    def _init_dataset(self):
        self.dataset = load_dataset(f"{self.config.DATA_PATH}/{self.config.DATA_LOAD_SCRIPT}",
                                    cache_dir=self.config.DATA_PATH)

        remove_path = glob.glob(f"{self.config.DATA_PATH}/**/TipoMensagem.java", recursive=True)
        if len(remove_path) > 0:
            os.remove(remove_path[0])

    def _init_tokenizer(self):
        vocab_path = os.path.join(self.config.CHECKPOINT_PATH, "vocab.json")
        if not os.path.exists(vocab_path):
            self._train_tokenizer()

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.CHECKPOINT_PATH,
                                                       max_len=512)
        print("Loaded tokenizer")

    def _train_tokenizer(self):
        paths = glob.glob(f"{self.config.DATA_PATH}/**/*.java", recursive=True)[:50000]
        print("Training tokenizer...")
        tok = ByteLevelBPETokenizer()
        tok.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
        ])
        tokenizer_path = self.config.CHECKPOINT_PATH
        if not os.path.exists(tokenizer_path):
            os.makedirs(tokenizer_path)
        tok.save_model(tokenizer_path)

        tokenizer_config = {
            "max_len": 512
        }

        with open(os.path.join(tokenizer_path, "config.json"), 'w') as fp:
            json.dump(tokenizer_config, fp)
        print(f"Trained tokenizer at {tokenizer_path}")

    def _get_model_config(self):
        if self.config.MODEL_NAME == "gpt":
            return AutoConfig.from_pretrained(
                "gpt2",
                vocab_size=len(self.tokenizer),
                n_ctx=self.config.CONTEXT_LENGTH,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        elif self.config.MODEL_NAME == "roberta":
            return RobertaConfig(
                vocab_size=52_000,
                max_position_embeddings=514,
                num_attention_heads=12,
                num_hidden_layers=6,
                type_vocab_size=1,
                n_ctx=self.config.CONTEXT_LENGTH,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

    def tokenize(self, element):
        outputs = self.tokenizer(
            element["text"],
            truncation=True,
            max_length=self.config.CONTEXT_LENGTH,
            return_overflowing_tokens=True,
            return_length=True,
        )
        return {"input_ids": outputs['input_ids']}

    def _load_model(self):
        print("Loading model...")
        model_config = self._get_model_config()
        if self.config.MODEL_NAME == "gpt":
            self.model = GPT2LMHeadModel(model_config)
        elif self.config.MODEL_NAME == "roberta":
            self.model = RobertaForMaskedLM(model_config)
        model_size = sum(t.numel() for t in self.model.parameters())
        print(f"Loaded model {self.config.MODEL_NAME} - {model_size / 1000 ** 2:.1f}M parameters")

    def train(self):
        self._init_tokenizer()
        self._load_model()

        tokenized_datasets = self.dataset.map(
            self.tokenize, batched=True, remove_columns=self.dataset["train"].column_names
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.config.MLM:
            data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=True, mlm_probability=0.15)
        else:
            data_collator = DataCollatorForLanguageModeling(self.tokenizer)
        set_seed()

        training_args = TrainingArguments(
            output_dir=self.config.CHECKPOINT_PATH,
            overwrite_output_dir=True,
            num_train_epochs=self.config.NUM_EPOCHS,
            per_device_train_batch_size=self.config.PER_DEVICE_BATCH_SIZE,
            per_device_eval_batch_size=self.config.PER_DEVICE_BATCH_SIZE,
            evaluation_strategy="steps",
            save_steps=self.config.SAVE_STEPS,
            eval_steps=self.config.EVAL_STEPS,
            save_total_limit=self.config.SAVE_LIMIT,
            # prediction_loss_only=True,
            # report_to="wandb",  # enable logging to W&B
            # run_name="codegpt2",
            learning_rate=self.config.LEARNING_RATE,
            fp16=self.config.FP_16,
        )
        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
        )

        trainer.train()
