from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
from textSummarizer.entity.entities import ModelTrainerConfig
import torch
import os
from textSummarizer.logging.logger import logger

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
    
    def train(self):
        logger.info(f"Entrada a main tokenizadores")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_t5 = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_t5)
        
        logger.info(f"carga de información de dataset")
        #loading data 
        dataset_samsum_pt = load_from_disk(self.config.data_path)
        logger.info(f"se llama a argumentos de entrenamiento")
        is_cuda = torch.cuda.is_available()
        """

        trainer_args = TrainingArguments(
             output_dir=self.config.root_dir,
             num_train_epochs=self.config.num_train_epochs,
             warmup_steps=self.config.warmup_steps,
             per_device_train_batch_size=self.config.per_device_train_batch_size,
             per_device_eval_batch_size=self.config.per_device_train_batch_size,
             weight_decay=self.config.weight_decay, logging_steps=self.config.logging_steps,
             eval_strategy=  self.config.evaluation_strategy, eval_steps=self.config.eval_steps, save_steps=self.config.save_steps,
             gradient_accumulation_steps=self.config.gradient_accumulation_steps
         ) 
        
                trainer_args = TrainingArguments(
            output_dir=self.config.root_dir, num_train_epochs=1, warmup_steps=50,
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            weight_decay=0.01, logging_steps=10,
            eval_strategy='steps', eval_steps=50, save_steps=50,
            gradient_accumulation_steps=16
        ) 
        """
        trainer_args = TrainingArguments(
                output_dir=self.config.root_dir,
                num_train_epochs=1,
                max_steps=50,                 # stop after 50 steps (remove later)
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=1,
                logging_steps=1,
                logging_first_step=True,
                report_to="none",             # no TB/W&B
                eval_strategy="no",     # disable eval (enable later)
                save_strategy="no",           # disable checkpoints (enable later)
                # platform specifics
                no_cuda=not is_cuda,
                fp16=is_cuda,                 # use FP16 if GPU supports it
                bf16=False,                   # set True only if your GPU supports bf16
                dataloader_pin_memory=is_cuda,
                dataloader_num_workers=0,     # safer on Windows
                # keep dataset columns intact for seq2seq
                remove_unused_columns=False,
                # reproducibility (optional)
                seed=42,
            )        

        
        logger.info(f"se llena la información para el trainer")
        trainer = Trainer(model=model_t5, args=trainer_args,
                  processing_class=tokenizer, data_collator=seq2seq_data_collator,
                  train_dataset=dataset_samsum_pt["train"], 
                  eval_dataset=dataset_samsum_pt["validation"])
        logger.info(f"se llama a entrenar al modelo con información de dataset")
        trainer.train()
        logger.info(f"guardar modelo")        
        ## Save model
        model_t5.save_pretrained(os.path.join(self.config.root_dir,"pegasus-samsum-model"))
        ## Save tokenizer
        logger.info(f"guardar tokenizador")
        tokenizer.save_pretrained(os.path.join(self.config.root_dir,"tokenizer"))