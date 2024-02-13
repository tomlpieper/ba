# from dataset_tool import DatasetTool
from openai import OpenAI
from loguru import logger
from BaseClassT5 import BaseClassT5
from transformers import T5Tokenizer, T5ForConditionalGeneration, default_data_collator, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch








def create_modified_dataset(
    splits: list, 
    amount_training_examples: int = None, 
    path: str = None
    ) -> None:
    """
    Create a modified dataset with rationale added to the original dataset.
    """
    for i in splits:
        df = tool.add_rationale_to_dataset(i, size=amount_training_examples)
        tool.write_dataset_to_json(split=i, path=path)

    logger.success(f"Successfully created modified dataset with rationale added to {splits}.")










if __name__ == "__main__":
    # client = OpenAI()
    # tool = DatasetTool(dataset_name='anli', client=client)
    splits = ['train_r1', 'dev_r1', 'test_r1', 'train_r2', 'dev_r2', 'test_r2', 'train_r3', 'dev_r3', 'test_r3']
    # logger.success(tool.get_dataset())
    # logger.success(splits[3:6])
    # logger.success(splits[6:])

    use_cuda = torch.cuda.is_available()
    print(f"Using GPU: {use_cuda}")

    # create_modified_dataset(splits[:3], amount_training_examples=100000, path='v1/full_r1/')
    # create_modified_dataset(splits[3:6], amount_training_examples=100000, path='v1/full_r2/')
    # create_modified_dataset(splits[6:], amount_training_examples=1000000, path='v1/full_r3/')
    # create_modified_dataset(['test_r1'], amount_training_examples=100, path='data/')
    args = Seq2SeqTrainingArguments(
                predict_with_generate=True,
                evaluation_strategy="steps",
                eval_steps=500,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                num_train_epochs=5,
                learning_rate=5e-5,
                output_dir="results/outputs",
                fp16=use_cuda,
                logging_dir="results/logs",
                logging_steps=5000
             # remove_unused_columns=False
            )
    model = BaseClassT5(
        model_name="t5-small",
        training_args=args
    )
    model.run(
        dataset_name="modified_anli", 
        splits=splits[:3],
        path_training_data="v1/full_r1/",
        # path_training_data="v1/data/",
        path_trained_model="v1/model",
        final_model_name="secondT5"
    )
    
    