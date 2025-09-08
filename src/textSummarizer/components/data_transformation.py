import os
from textSummarizer.logging.logger import logger
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
from textSummarizer.entity.entities import DataTransformationConfig



class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)


    
    def convert_examples_to_features(self,example_batch):
        inputs = ["summarize: " + doc for doc in example_batch["dialogue"]]
        input_encodings = self.tokenizer( inputs , max_length = 256, truncation = True )
        
        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(example_batch['summary'], max_length = 64, truncation = True)
        
        input_encodings["labels"] = target_encodings["input_ids"]
        
        return input_encodings
    """
    {
            'input_ids' : input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }
    """

    def convert(self):
        dataset_samsum = load_from_disk(self.config.data_path)
        keep_cols = ["input_ids","attention_mask","labels","dialogue","summary"]
        dataset_samsum_pt = dataset_samsum.map(self.convert_examples_to_features,
                                               batched = True, remove_columns=dataset_samsum["train"].column_names)
        dataset_samsum_pt.save_to_disk(os.path.join(self.config.root_dir,"samsum_dataset"))

