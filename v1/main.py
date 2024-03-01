# from dataset_tool import DatasetTool
from openai import OpenAI
from loguru import logger
from BaseClassT5 import BaseClassT5
from transformers import T5Tokenizer, T5ForConditionalGeneration, default_data_collator, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch
import mlflow




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



def create_paths(model: str, lr: float, split_ratio: tuple) -> tuple:
    """
    Create the paths for the model weights and logs.

    Args:
        model (str): Model name.
        lr (float): Learning rate.
        split_ratio (tuple): Split ratio.

    Returns:
        2 Strings, logging_path and weights_path: Paths for the model weights and logs.
    """
    if split_ratio != None:
        lr_str = str(lr)
        split_ratio_str = str(split_ratio)
        logging_dir = "logs/"
        weights_dir = "results/"
        weights_path = weights_dir + model + "-weights/" + lr_str + "/" + split_ratio_str + "/"
        logging_path =  logging_dir + model + "-logs/" + lr_str + "_" + split_ratio_str + "/"
    else:
        lr_str = str(lr)
        logging_dir = "logs/"
        weights_dir = "results/"
        weights_path = weights_dir + model + "-weights/" + lr_str + "/"
        logging_path =  logging_dir + model + "-logs/" + lr_str + "/"
    return logging_path, weights_path




def run_different_lrs(split_ratios: list, lr: float = 3e-4):
    """
    Run the model with different learning rates and split ratios.

    Args:
        split_ratios (list): List of split ratios to use.
        lr (float, optional): Learning rate for training. Defaults to 3e-4.
    """


    for i in split_ratio:
        logging_path, weights_path = create_paths("t5-small", lr, i)
        run_modified_anli_with_rationale(
            splits=splits, 
            split_ratio=i, 
            lr=lr, 
            use_cuda=True, 
            train_batch_size=8, 
            eval_batch_size=8, 
            logging_path=logging_path, 
            weights_path=weights_path
        )


def run_modified_anli_with_rationale(
    splits: [str],
    split_ratio: tuple,
    lr: float = 3e-4,
    use_cuda: bool = True,
    train_batch_size: int = 8,
    eval_batch_size: int = 8,
    logging_path: str = None,
    weights_path: str = None,
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
    logging_path, weights_path = create_paths(model_name, lr, split_ratio)

    args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        generation_max_length=400,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        warmup_steps=500,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=5,
        learning_rate=lr,
        output_dir= weights_path + "outputs",
        fp16=use_cuda,
        logging_dir=weights_path + "logs",
        logging_steps=100,
        load_best_model_at_end=True,     # Load the best model when finished training (default metric is loss)
        metric_for_best_model='label_accuracy', # Use accuracy as the best metric unless it is baselineModel then use exact_match
    )

    model = BaseClassT5(
        model_name=model_name,
        training_args=args,
        path_custom_logs=logging_path,
        path_model_weights=weights_path,
        model_type=0,
        ratio=split_ratio
    )

    model.run(
        dataset_name="modified_anli", 
        splits=splits,
        path_training_data="v1/full_r1/",
        path_trained_model=weights_path,
        final_model_name=model_name
    )


def run_original_anli_with_rationale(
    splits: [str], 
    split_ratio: tuple, 
    lr: float = 3e-4, 
    use_cuda: bool = True, 
    train_batch_size: int = 8, 
    eval_batch_size: int = 8, 
    logging_path: str = None, 
    weights_path: str = None,
    model_name: str = "t5-small"
    ):
    """
    Run the original ANLI dataset with rationale added to the original dataset.
    
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
    logging_path, weights_path = create_paths(model_name, lr, split_ratio)

    args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        generation_max_length=400,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        warmup_steps=500,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=5,
        learning_rate=lr,
        output_dir= weights_path + "outputs",
        fp16=use_cuda,
        logging_dir=weights_path + "logs",
        logging_steps=100,
        load_best_model_at_end=True,     # Load the best model when finished training (default metric is loss)
        metric_for_best_model='label_accuracy', # Use accuracy as the best metric unless it is baselineModel then use exact_match
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
        path_training_data="v1/full_r1/",
        path_trained_model=weights_path,
        final_model_name=model_name
    )


def run_orignal_anli_without_rationale(
    splits: [str], 
    lr: float = 3e-4, 
    use_cuda: bool = True, 
    train_batch_size: int = 8, 
    eval_batch_size: int = 8, 
    logging_path: str = None, 
    weights_path: str = None,
    model_name: str = "t5-small"
    ):

    """
    Run the original ANLI dataset without rationale added to the original dataset.
    
    Args:
        splits (list[str]): List of dataset splits to use.
        lr (float, optional): Learning rate for training. Defaults to 3e-4.
        use_cuda (bool, optional): Whether to use CUDA for training. Defaults to True.
        train_batch_size (int, optional): Batch size for training. Defaults to 8.
        eval_batch_size (int, optional): Batch size for evaluation. Defaults to 8.
        logging_path (str, optional): Path to save logs. Defaults to None.
        weights_path (str, optional): Path to save model weights. Defaults to None.
    """
    logging_path, weights_path = create_paths(model_name, lr, None)

    args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        generation_max_length=400,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        warmup_steps=500,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=5,
        learning_rate=lr,
        output_dir= weights_path + "outputs",
        fp16=use_cuda,
        logging_dir=weights_path + "logs",
        logging_steps=100,
        load_best_model_at_end=True,     # Load the best model when finished training (default metric is loss)
        metric_for_best_model='exact_match_accuracy', # Use accuracy as the best metric unless it is baselineModel then use exact_match
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
        path_training_data="v1/full_r1/",
        path_trained_model=weights_path,
        final_model_name=model_name
    )


if __name__ == "__main__":
    # client = OpenAI()
    # tool = DatasetTool(dataset_name='anli', client=client)
    splits = ['train_r1', 'dev_r1', 'test_r1', 'train_r2', 'dev_r2', 'test_r2', 'train_r3', 'dev_r3', 'test_r3']
    # logger.success(tool.get_dataset())
    # logger.success(splits[3:6])
    # logger.success(splits[6:])
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    use_cuda = torch.cuda.is_available()
    print(f"Using GPU: {use_cuda}")
    train_batch_size = 8 if use_cuda else 1
    eval_batch_size = 8 if use_cuda else 1


    # create_modified_dataset(splits[:3], amount_training_examples=100000, path='v1/full_r1/')
    # create_modified_dataset(splits[3:6], amount_training_examples=100000, path='v1/full_r2/')
    # create_modified_dataset(splits[6:], amount_training_examples=1000000, path='v1/full_r3/')
    # create_modified_dataset(['test_r1'], amount_training_examples=100, path='data/')
    lr = 3e-4
    lr_str = str(lr)
    # split_ratio = (0.25, 0.75)
    split_ratio = (0.75, 0.25)
    split_ratio_str = str(split_ratio)



    # run_different_lrs(
    #     split_ratios=[(0.25, 0.75),(0.5,0.5), (0.75, 0.25)],
    #     lr=3e-4
    # )
# run_orignal_anli_without_rationale(
#     splits=splits[:3],
#     use_cuda=use_cuda,
#     train_batch_size=train_batch_size,
#     eval_batch_size=eval_batch_size
# )
run_modified_anli_with_rationale(
    splits=splits[:3],
    split_ratio=(0.5,0.5),
    use_cuda=use_cuda
    
)


model_types = {
    0: "modified_anli_with_rationale", 
    1: "original_anli_with_rationale", 
    2: "orignal_anli_without_rationale"}
