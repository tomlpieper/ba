from v1.dataset_tool import DatasetTool
import os
import openai
 
openai.api_key = os.environ["OPENAI_API_KEY"]
from openai import OpenAI
from loguru import logger
# from v1.BaseClassT5 import BaseClassT5
# from transformers import T5Tokenizer, T5ForConditionalGeneration, default_data_collator, Seq2SeqTrainingArguments, Seq2SeqTrainer
# import torch
# import mlflow
import os



def create_modified_dataset(
    splits: list, 
    amount_training_examples: int = None, 
    path: str = None
    ) -> None:
    """
    Create a modified dataset with rationale added to the original dataset.
    """
    os.makedirs(path, exist_ok=True)
    for i in splits:
        df = tool.add_rationale_to_dataset(i, size=amount_training_examples)
        tool.write_dataset_to_json(split=i, path=path)

    logger.success(f"Successfully created modified dataset with rationale added to {splits}.")




if __name__ == "__main__":
    client = OpenAI(organization='org-U1REoyfP2aBfZm9zTzz5uWUo')
    tool = DatasetTool(dataset_name='anli', client=client)
    splits = ['train_r1', 'dev_r1', 'test_r1', 'train_r2', 'dev_r2', 'test_r2', 'train_r3', 'dev_r3', 'test_r3']
    # logger.success(tool.get_dataset())
    # logger.success(splits[3:6])
    # logger.success(splits[6:])
    # train_batch_size = 8 if use_cuda else 1
    # eval_batch_size = 8 if use_cuda else 1

    create_modified_dataset(splits[:3], amount_training_examples=100000, path='/Users/tompieper/code_3/v2/full_r1/')
    # create_modified_dataset(splits[3:6], amount_training_examples=100000, path='v1/full_r2/')
    # create_modified_dataset(splits[6:], amount_training_examples=1000000, path='v1/full_r3/')
    # create_modified_dataset(['test_r1'], amount_training_examples=100, path='data/')
    # lr = 3e-4