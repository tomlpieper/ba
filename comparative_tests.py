# import warnings
# warnings.simplefilter("ignore", UserWarning)
# from datasets import load_dataset, Dataset
import pandas as pd
import numpy as np
# from transformers import (
#     T5Tokenizer, 
#     T5ForConditionalGeneration, 
#     default_data_collator, 
#     Seq2SeqTrainingArguments, 
#     Seq2SeqTrainer, 
#     TrainerCallback, 
#     AutoModelForSeq2SeqLM, 
#     AutoTokenizer, 
#     EarlyStoppingCallback
#     )
# from transformers.integrations import MLflowCallback
# from loguru import logger
# import random
import json
# import torch
import os
# from sklearn.metrics import f1_score, precision_score, recall_score
# import matplotlib.pyplot as plt
# from copy import deepcopy
# from loguru import logger
# from CustomTrainer import CustomTrainer
# from callbacks import CustomCallback
# from v1.BaseClassT5 import BaseClassT5

# from nltk.translate.bleu_score import sentence_bleu







file_paths_metrics = {
    "baselines": [
        "/netscratch/tpieper/v3/baseline/logs/t5-small-logs/0.0006/metrics.json",
        "/netscratch/tpieper/v3/baseline/logs/t5-small-logs/0.0012/metrics.json",
        "/netscratch/tpieper/v3/baseline/logs/t5-small-logs/0.0024/metrics.json"
    ],
    "modified": [
        "/netscratch/tpieper/v2logs/t5-small-logs/0.0006_(0.25, 0.75)/metrics.json",
        "/netscratch/tpieper/v2logs/t5-small-logs/0.0012_(0.25, 0.75)/metrics.json",
        "/netscratch/tpieper/v2logs/t5-small-logs/0.0024_(0.25, 0.75)/metrics.json",
        "/netscratch/tpieper/v2logs/t5-small-logs/0.0006_(0.5, 0.5)/metrics.json",
        "/netscratch/tpieper/v2logs/t5-small-logs/0.0012_(0.5, 0.5)/metrics.json",
        "/netscratch/tpieper/v2logs/t5-small-logs/0.0024_(0.5, 0.5)/metrics.json",
        "/netscratch/tpieper/v2logs/t5-small-logs/0.0006_(0.75, 0.25)/metrics.json",
        "/netscratch/tpieper/v2logs/t5-small-logs/0.0012_(0.75, 0.25)/metrics.json",
        "/netscratch/tpieper/v2logs/t5-small-logs/0.0024_(0.75, 0.25)/metrics.json"
    ],
    "modified_2": [
        "/netscratch/tpieper/v3/logs/t5-small-logs/0.0006_(0.25, 0.75)/metrics.json",
        "/netscratch/tpieper/v3/logs/t5-small-logs/0.0012_(0.25, 0.75)/metrics.json",
        "/netscratch/tpieper/v3/logs/t5-small-logs/0.0024_(0.25, 0.75)/metrics.json",
        "/netscratch/tpieper/v3/logs/t5-small-logs/0.0006_(0.5, 0.5)/metrics.json",
        "/netscratch/tpieper/v3/logs/t5-small-logs/0.0012_(0.5, 0.5)/metrics.json",
        "/netscratch/tpieper/v3/logs/t5-small-logs/0.0024_(0.5, 0.5)/metrics.json",
        "/netscratch/tpieper/v3/logs/t5-small-logs/0.0006_(0.75, 0.25)/metrics.json",
        "/netscratch/tpieper/v3/logs/t5-small-logs/0.0012_(0.75, 0.25)/metrics.json",
        "/netscratch/tpieper/v3/logs/t5-small-logs/0.0024_(0.75, 0.25)/metrics.json"
    ]
}

file_paths_weights = {
    "baselines": [
        "/netscratch/tpieper/v3/baseline/results/t5-small-weights/0.0006/outputs/",
        "/netscratch/tpieper/v3/baseline/results/t5-small-weights/0.0012/outputs/",
        "/netscratch/tpieper/v3/baseline/results/t5-small-weights/0.0024/outputs/"
    ],
    "modified": [
        "/netscratch/tpieper/v2/results/t5-small-weights/0.0006/(0.25, 0.75)/outputs",
        "/netscratch/tpieper/v2/results/t5-small-weights/0.0012/(0.25, 0.75)/outputs",
        "/netscratch/tpieper/v2/results/t5-small-weights/0.0024/(0.25, 0.75)/outputs",
        "/netscratch/tpieper/v3/results/t5-small-weights/0.0006/(0.5, 0.5)/outputs",
        "/netscratch/tpieper/v3/results/t5-small-weights/0.0012/(0.5, 0.5)/outputs",
        "/netscratch/tpieper/v3/results/t5-small-weights/0.0024/(0.5, 0.5)/outputs",
        "/netscratch/tpieper/v3/results/t5-small-weights/0.0006/(0.75, 0.25)/outputs",
        "/netscratch/tpieper/v3/results/t5-small-weights/0.0012/(0.75, 0.25)/outputs",
        "/netscratch/tpieper/v3/results/t5-small-weights/0.0024/(0.75, 0.25)/outputs"

    ],
    "modified_2": [
        "/netscratch/tpieper/v3/results/t5-small-weights/0.0006/(0.25, 0.75)/outputs",
        "/netscratch/tpieper/v3/results/t5-small-weights/0.0012/(0.25, 0.75)/outputs",
        "/netscratch/tpieper/v3/results/t5-small-weights/0.0024/(0.25, 0.75)/outputs",
        "/netscratch/tpieper/v3/results/t5-small-weights/0.0006/(0.5, 0.5)/outputs",
        "/netscratch/tpieper/v3/results/t5-small-weights/0.0012/(0.5, 0.5)/outputs",
        "/netscratch/tpieper/v3/results/t5-small-weights/0.0024/(0.5, 0.5)/outputs",
        "/netscratch/tpieper/v3/results/t5-small-weights/0.0006/(0.75, 0.25)/outputs",
        "/netscratch/tpieper/v3/results/t5-small-weights/0.0012/(0.75, 0.25)/outputs",
        "/netscratch/tpieper/v3/results/t5-small-weights/0.0024/(0.75, 0.25)/outputs"
    ]

}



def check_access():

    # verify that all files in the file_paths_metrics exist 
    for key, value in file_paths_metrics.items():
        for file in value:
            if not os.path.exists(file):
                raise ValueError(f"File {file} does not exist")

    # check that all of the folders in file_paths_weights exist
    for key, value in file_paths_weights.items():
        for folder in value:
            if not os.path.exists(folder):
                raise ValueError(f"Folder {folder} does not exist")

# here we can use steps as anker point for the checkpoints

def find_checkpoint_with_best_label_accuracy(file_path: str):
    with open(file_path, 'r') as file:

        best = 0
        epoch = None
        data = json.load(file)
        for i in data['log_history']:
            if not 'eval_label_accuracy' in i:
                pass
            elif i['eval_label_accuracy'] > best:
                best = i['eval_label_accuracy']
                epoch = i['epoch']
                step = i['step']
                checkpoint = "checkpoint-" + str(step)
            else:
                pass
        return best, epoch, checkpoint
        # max_label_accuracy = max([entry['eval_label_accuracy'] for entry in data['log_history']])
        # best_checkpoint = [entry['checkpoint'] for entry in data['log_history'] if entry['eval_label_accuracy'] == max_label_accuracy]
        # return best_checkpoint[0]

# def test_model(checkpoint_path: str, test_data: Dataset):
    
# Combine both dictionaries into a list of tuples for DataFrame construction
data = []
for category, paths in file_paths_metrics.items():
    for path in paths:
        config = path.split('/')[-2]
        name = category
        data_type = 'metrics'
        data.append((name, config, data_type, path))

for category, paths in file_paths_weights.items():
    for path in paths:
        config = path.split('/')[-3]
        name = category
        data_type = 'weights'
        data.append((name, config, data_type, path))

# Create DataFrame
df = pd.DataFrame(data, columns=['Name', 'Config', 'Type', 'Path'])
print(df)


test_path = '/Users/tompieper/new/t5-small-logs/0.0006_(0.5, 0.5)/metrics.json'

print(find_checkpoint_with_best_label_accuracy(test_path))


# Merge complete lists into pd.Dataframe 
# df = pd.DataFrame(list(zip(file_paths_metrics, file_paths_weights)), columns=['metrics', 'weights'])
# print(df.head())


# create a Dataframe with all the metrics and weights and a column for their names
df = pd.DataFrame(list(zip(file_paths_metrics, file_paths_weights)), columns=['metrics', 'weights'])
print(df.head())















# The Checkpoint to load is given to the model name parameter
# If run on V2, V2 bool has to be set to True as well as the correct path for the training dataset has to be set
# Call this for the modfiied dataset 
# model_type = 0, v2 = False for the modified run
# 
# 
# Call this for the modified dataset V2 
# model_type = 0, v2 = True for the modified run
# 
# Call this for the original dataset
# model_type = 2, v2 = False for the original run



# prepare test dataset using the T5Baseclass
def test_on_best_label_accuracy(
    splits: ["train_r1", "dev_r1", "test_r1"],
    lr: float = 6e-4,
    epochs: int = 5,
    use_cuda: bool = True,
    train_batch_size: int = 32,
    eval_batch_size: int = 32,
    eval_steps: int = 132,
    base_path: str = "/netscratch/tpieper/tests/",
    data_path: str = "/netscratch/tpieper/v1/full_r1/",
    model_name: str = "t5-small",
    v2: bool = False
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
    # logging_path, weights_path = create_paths(base_path, model_name, lr, None)
    if v2:
        data_path = "/netscratch/tpieper/v2/full_r1/"
    else:
        data_path = "/netscratch/tpieper/v1/full_r1/"



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

    
    if model_type == 2:
        model = BaseClassT5(
        model_name=model_name,
        training_args=args,
        path_custom_logs=logging_path,
        path_model_weights=weights_path,
        model_type=2
        )
        
        model.load_and_process_dataset(dataset_name=dataset_name, splits=splits)

        model.trainer = CustomTrainer(
                model=model,
                args=args,
                train_dataset=model.train_split,
                eval_dataset=model.dev_split,
                data_collator=default_data_collator,
                compute_metrics=model.compute_exact_match,
                split_loss=model.split_loss,
                ratio=model.ratio
            )
        results = model.trainer.evaluate(model.test_split, metric_key_prefix="test")
        print(results)
        logger.success(f"Test results: {results}")
        print(model.trainer.state)

    elif model_type == 0:
        model = BaseClassT5(
        model_name=model_name,
        training_args=args,
        path_custom_logs=logging_path,
        path_model_weights=weights_path,
        model_type=2,
        v2=v2
        )
        model.trainer = CustomTrainer(
                model=model,
                args=args,
                train_dataset=model.train_split,
                eval_dataset=model.dev_split,
                data_collator=default_data_collator,
                compute_metrics=model.compute_metrics,
                split_loss=model.split_loss,
                ratio=model.ratio
            )
        model.load_local_dataset(dataset_name=dataset_name, splits=splits, path=path_training_data)
        model.prepare_training()
        results = model.trainer.evaluate(model.test_split, metric_key_prefix="test")
        print(results)




    else:
        raise ValueError("Model type must be either 0 or 2 for evaluating on test data.")


    return results
