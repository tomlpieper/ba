import warnings
warnings.simplefilter("ignore", UserWarning)
from datasets import load_dataset, Dataset
import pandas as pd
import numpy as np
from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration, 
    default_data_collator, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
    TrainerCallback, 
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    EarlyStoppingCallback
    )
from transformers.integrations import MLflowCallback
from loguru import logger
import random
import json
import torch
import os
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from copy import deepcopy
from loguru import logger
from CustomTrainer import CustomTrainer
from callbacks import CustomCallback
from BaseClassT5 import BaseClassT5
from wrapped_runs import create_paths

from nltk.translate.bleu_score import sentence_bleu







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
        "/netscratch/tpieper/v3/baseline/results/t5-small-weights/0.0006/outputs",
        "/netscratch/tpieper/v3/baseline/results/t5-small-weights/0.0012/outputs",
        "/netscratch/tpieper/v3/baseline/results/t5-small-weights/0.0024/outputs"
    ],
    "modified": [
        "/netscratch/tpieper/v2results/t5-small-weights/0.0006/(0.25, 0.75)/outputs",
        "/netscratch/tpieper/v2results/t5-small-weights/0.0012/(0.25, 0.75)/outputs",
        "/netscratch/tpieper/v2results/t5-small-weights/0.0024/(0.25, 0.75)/outputs",
        "/netscratch/tpieper/v2results/t5-small-weights/0.0006/(0.5, 0.5)/outputs",
        "/netscratch/tpieper/v2results/t5-small-weights/0.0012/(0.5, 0.5)/outputs",
        "/netscratch/tpieper/v2results/t5-small-weights/0.0024/(0.5, 0.5)/outputs",
        "/netscratch/tpieper/v2results/t5-small-weights/0.0006/(0.25, 0.75)/outputs",
        "/netscratch/tpieper/v2results/t5-small-weights/0.0012/(0.25, 0.75)/outputs",
        "/netscratch/tpieper/v2results/t5-small-weights/0.0024/(0.25, 0.75)/outputs"

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
def jls_extract_def():
    # Combine the two lists into a dataframe
    df = pd.DataFrame({
        'Type': ['baselines'] * len(file_paths_metrics['baselines']) + 
            ['modified'] * len(file_paths_metrics['modified']) + 
            ['modified_2'] * len(file_paths_metrics['modified_2']),
        'Metrics': file_paths_metrics['baselines'] + 
               file_paths_metrics['modified'] + 
               file_paths_metrics['modified_2'],
        'Weights': file_paths_weights['baselines'] + 
               file_paths_weights['modified'] + 
               file_paths_weights['modified_2']
    })
    
    # Extract learning rate and split ratio from file paths
    df['Config'] = df['Metrics'].apply(lambda path: (path.split('/')[-2]))
        
    df['Best_Checkpoint'] = df['Metrics'].apply(find_checkpoint_with_best_label_accuracy)
    df['Model_Checkpoint_Best_Label_Accuracy'] = df['Weights'] + '/' + df['Best_Checkpoint']
    
    return df





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
    print("Found all necessary files and folders")

# here we can use steps as anker point for the checkpoints

def find_checkpoint_with_best_label_accuracy(file_path: str):
    with open(file_path, 'r') as file:
        best_label_accuracy = 0
        best_exact_match_accuracy = 0
        epoch = None
        checkpoint = None
        data = json.load(file)
        for i in data['log_history']:
            if 'eval_label_accuracy' in i:
                if i['eval_label_accuracy'] > best_label_accuracy:
                    best_label_accuracy = i['eval_label_accuracy']
                    epoch = i['epoch']
                    step = i['step']
                    checkpoint = "checkpoint-" + str(step)
            elif 'eval_exact_match_accuracy' in i:
                if i['eval_exact_match_accuracy'] > best_exact_match_accuracy:
                    best_exact_match_accuracy = i['eval_exact_match_accuracy']
                    epoch = i['epoch']
                    step = i['step']
                    checkpoint = "checkpoint-" + str(step)
        return checkpoint


def get_test_results(df):
    results = []
    for index, row in df.iterrows():
        print(f"Running test {index+1} of {len(df)}")
        if row['Type'] == 'baselines':
            result = test_on_best_label_accuracy(
                model_name=row['Model_Checkpoint_Best_Label_Accuracy'],
                v2=False,
                model_type=2,
            )
        elif row['Type'] == 'modified':
            result = test_on_best_label_accuracy(
                model_name=row['Model_Checkpoint_Best_Label_Accuracy'],
                v2=False,
                model_type=0,
    
            )
        elif row['Type'] == 'modified_2':
            result = test_on_best_label_accuracy(
                model_name=row['Model_Checkpoint_Best_Label_Accuracy'],
                v2=True,
                model_type=0,
            )
        
        if 'test_label_accuracy' in result:
            df.at[index, 'Results'] = result['test_label_accuracy']
        elif 'test_exact_match_accuracy' in result:
            df.at[index, 'Results'] = result['test_exact_match_accuracy']
        else:
            raise ValueError("No test accuracy found in results")
    
    return df


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
    splits: [str] = ["train_r1", "dev_r1", "test_r1"],
    lr: float = 6e-4,
    epochs: int = 5,
    use_cuda: bool = True,
    train_batch_size: int = 32,
    eval_batch_size: int = 32,
    eval_steps: int = 132,
    base_path: str = "/netscratch/tpieper/tests/",
    data_path: str = "/netscratch/tpieper/v1/full_r1/",
    model_name: str = "t5-small",
    v2: bool = False,
    model_type: int = 0
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
        model_type=2,
        dataset_name="anli",
        )
        
        model.load_and_process_dataset(dataset_name=model.dataset_name, splits=splits)

        model.trainer = CustomTrainer(
                model=model.model,
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
        v2=v2,
        dataset_name="modified_anli",
        )
        model.trainer = CustomTrainer(
                model=model.model,
                args=args,
                train_dataset=model.train_split,
                eval_dataset=model.dev_split,
                data_collator=default_data_collator,
                compute_metrics=model.compute_metrics,
                split_loss=model.split_loss,
                ratio=model.ratio
            )
        model.load_local_dataset(dataset_name=model.dataset_name, splits=splits, path=data_path)
        model.prepare_training()
        results = model.trainer.evaluate(model.test_split, metric_key_prefix="test")
        print(results)




    else:
        raise ValueError("Model type must be either 0 or 2 for evaluating on test data.")


    return results


if __name__ == "__main__":


    check_access()
    df = jls_extract_def()
    results = get_test_results(df)
    print(results)
    # results.to_csv(path + "alterntive_test_results.csv")
    # write to json
    results.to_json("alternative_test_results.json")











# test_path = '/Users/tompieper/new/t5-small-logs/0.0006_(0.5, 0.5)/metrics.json'
# test_path_2 = '/Users/tompieper/metrics.json'

# print(find_checkpoint_with_best_label_accuracy(test_path))
# print(find_checkpoint_with_best_label_accuracy(test_path_2))



