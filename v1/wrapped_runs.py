from openai import OpenAI
from loguru import logger
from BaseClassT5 import BaseClassT5
from transformers import T5Tokenizer, T5ForConditionalGeneration, default_data_collator, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch
import mlflow
import os

# torch.distributed.init_process_group('NCCL')
# local_rank = int(os.environ['LOCAL_RANK'])
# torch.cuda.set_device(local_rank)
# device = torch.device("cuda", local_rank)
# cuda_visible = os.environ['CUDA_VISIBLE_DEVICES']




def create_paths(base_path: str, model: str, lr: float, split_ratio: tuple) -> tuple:
    """
    Create the paths for the logging and weights directories

    Args:
        base_path: str: The base path for the project
        model: str: The model name
        lr: float: The learning rate
        split_ratio: tuple: The split ratio for the dataset

    """
    if split_ratio != None:
        lr_str = str(lr)
        split_ratio_str = str(split_ratio)

        weights_path =  model + "-weights/" + lr_str + "/" + split_ratio_str + "/"
        logging_path =  model + "-logs/" + lr_str + "_" + split_ratio_str + "/"
    else:
        lr_str = str(lr)
        weights_path =  model + "-weights/" + lr_str + "/"
        logging_path =  model + "-logs/" + lr_str + "/"

    logging_dir = "logs/"
    weights_dir = "results/"

    logging_path = base_path + logging_dir + logging_path
    weights_path = base_path + weights_dir + weights_path

    return logging_path, weights_path


def run_modified_anli_with_rationale(
    splits: [str],
    split_ratio: tuple,
    lr: float = 3e-4,
    epochs: int = 5,
    use_cuda: bool = True,
    train_batch_size: int = 8,
    eval_batch_size: int = 8,
    eval_steps: int = 66,
    base_path: str = "/netscratch/tpieper/",
    data_path: str = "/netscratch/tpieper/v1/full_r1/",
    model_name: str = "t5-small"
    ):
    """
    Run the modified ANLI dataset with rationale added to the original dataset.

    Args:
        splits (list[str]): List of dataset splits to use.
        split_ratio (tuple): Tuple containing the ratio of original dataset to rationale dataset.
        lr (float, optional): Learning rate for training. Defaults to 3e-4.
        use_cuda (bool, optional): Whether to use CUDA for training. Defaults to True.
        train_batch_size (int, optional): Batch size for training. Defaults to 8.
        eval_batch_size (int, optional): Batch size for evaluation. Defaults to 8.
        logging_path (str, optional): Path to save logs. Defaults to None.
        weights_path (str, optional): Path to save model weights. Defaults to None.
    """
    logging_path, weights_path = create_paths(base_path, model_name, lr, split_ratio)

    logging_steps = eval_steps/2

    args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        generation_max_length=400,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_steps=eval_steps,
        warmup_steps=100,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=epochs,
        learning_rate=lr,
        output_dir= weights_path + "outputs",
        logging_dir=weights_path + "logs",
        logging_steps=logging_steps,
        load_best_model_at_end=True,     # Load the best model when finished training (default metric is loss)
        # metric_for_best_model='label_accuracy'
    )

    model = BaseClassT5(
        model_name=model_name,
        training_args=args,
        path_custom_logs=logging_path,
        path_model_weights=weights_path,
        model_type=0,
        v2=True,
        ratio=split_ratio
    )

    model.run(
        dataset_name="modified_anli", 
        splits=splits,
        path_training_data=data_path,
        path_trained_model=weights_path,
        final_model_name=model_name
    )




def run_original_anli_with_rationale(
    splits: [str],
    split_ratio: tuple,
    lr: float = 3e-4,
    epochs: int = 5,
    use_cuda: bool = True,
    train_batch_size: int = 8,
    eval_batch_size: int = 8,
    eval_steps: int = 66,
    base_path: str = "/netscratch/tpieper/",
    data_path: str = "/netscratch/tpieper/v1/full_r1/",
    model_name: str = "t5-small"
    ):
    """
    Run the modified ANLI dataset with rationale added to the original dataset.

    Args:
        splits (list[str]): List of dataset splits to use.
        split_ratio (tuple): Tuple containing the ratio of original dataset to rationale dataset.
        lr (float, optional): Learning rate for training. Defaults to 3e-4.
        use_cuda (bool, optional): Whether to use CUDA for training. Defaults to True.
        train_batch_size (int, optional): Batch size for training. Defaults to 8.
        eval_batch_size (int, optional): Batch size for evaluation. Defaults to 8.
        logging_path (str, optional): Path to save logs. Defaults to None.
        weights_path (str, optional): Path to save model weights. Defaults to None.
    """
    logging_path, weights_path = create_paths(base_path, model_name, lr, split_ratio)

    logging_steps = eval_steps/2

    args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        generation_max_length=400,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_steps=eval_steps,
        warmup_steps=100,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=epochs,
        learning_rate=lr,
        output_dir= weights_path + "outputs",
        logging_dir=weights_path + "logs",
        logging_steps=logging_steps,
        load_best_model_at_end=True,     # Load the best model when finished training (default metric is loss)
        # metric_for_best_model='label_accuracy'
    )

    model = BaseClassT5(
        model_name=model_name,
        training_args=args,
        path_custom_logs=logging_path,
        path_model_weights=weights_path,
        model_type=1,
        ratio=split_ratio
    )

    model.run(
        dataset_name="anli", 
        splits=splits,
        path_training_data=data_path,
        path_trained_model=weights_path,
        final_model_name=model_name
    )




def run_original_anli_without_rationale(
    splits: [str],
    lr: float = 3e-4,
    epochs: int = 5,
    use_cuda: bool = True,
    train_batch_size: int = 8,
    eval_batch_size: int = 8,
    eval_steps: int = 66,
    base_path: str = "/netscratch/tpieper/",
    data_path: str = "/netscratch/tpieper/v1/full_r1/",
    model_name: str = "t5-small"
    ):
    """
    Run the modified ANLI dataset with rationale added to the original dataset.

    Args:
        splits (list[str]): List of dataset splits to use.
        split_ratio (tuple): Tuple containing the ratio of original dataset to rationale dataset.
        lr (float, optional): Learning rate for training. Defaults to 3e-4.
        use_cuda (bool, optional): Whether to use CUDA for training. Defaults to True.
        train_batch_size (int, optional): Batch size for training. Defaults to 8.
        eval_batch_size (int, optional): Batch size for evaluation. Defaults to 8.
        logging_path (str, optional): Path to save logs. Defaults to None.
        weights_path (str, optional): Path to save model weights. Defaults to None.
    """
    logging_path, weights_path = create_paths(base_path, model_name, lr, None)

    logging_steps = eval_steps/2

    args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        generation_max_length=400,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_steps=eval_steps,
        warmup_steps=100,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=epochs,
        learning_rate=lr,
        output_dir= weights_path + "outputs",
        logging_dir=weights_path + "logs",
        logging_steps=logging_steps,
        load_best_model_at_end=True,     # Load the best model when finished training (default metric is loss)
        # metric_for_best_model='exact_match_accuracy'
    )

    model = BaseClassT5(
        model_name=model_name,
        training_args=args,
        path_custom_logs=logging_path,
        path_model_weights=weights_path,
        model_type=2
    ) 

    model.run(
        dataset_name="anli", 
        splits=splits,
        path_training_data=data_path,
        path_trained_model=weights_path,
        final_model_name=model_name
    )
