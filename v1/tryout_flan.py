# from dataset_tool import DatasetTool
from openai import OpenAI
from loguru import logger
from BaseClassT5 import BaseClassT5
from transformers import T5Tokenizer, T5ForConditionalGeneration, default_data_collator, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoModelForSeq2SeqLM, AutoTokenizer
import torch




if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    print(f"Using GPU: {use_cuda}")
    train_batch_size = 8 if use_cuda else 1
    eval_batch_size = 8 if use_cuda else 1

    splits = ['train_r1', 'dev_r1', 'test_r1', 'train_r2', 'dev_r2', 'test_r2', 'train_r3', 'dev_r3', 'test_r3']
    result_dir = "results/"
    logging_path =  result_dir + "flan-t5-small/"

    args = Seq2SeqTrainingArguments(
                predict_with_generate=True,
                evaluation_strategy="steps",
                eval_steps=50,
                per_device_train_batch_size=train_batch_size,
                per_device_eval_batch_size=eval_batch_size,
                num_train_epochs=5,
                learning_rate=5e-5,
                output_dir= logging_path + "outputs",
                fp16=use_cuda,
                logging_dir=logging_path + "logs",
                logging_steps=50
             # remove_unused_columns=False
            )
    model = BaseClassT5(
        model_name="google/flan-t5-small",
        training_args=args,
        path_custom_logs=logging_path,
        baseline_model=False,
        flan=True

    )
    model.run(
        dataset_name="modified_anli", 
        splits=splits[:3],
        path_training_data="v1/full_r1/",
        # path_training_data="v1/data/",
        path_trained_model="v1/model3/",
        final_model_name="flan-t5-small"
    )
