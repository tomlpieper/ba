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

from nltk.translate.bleu_score import sentence_bleu
import numpy as np



class BaseClassT5:   
    
    def __init__(self, 
        model_name: str = "t5-base", 
        training_args: Seq2SeqTrainingArguments = None, 
        path_custom_logs: str = "results", 
        model_type: int = 0,
        path_model_weights: str = 'results', 
        ratio: tuple = (0.5,0.5) 
        ) -> None:
            """
            Initialize the BaseClassT5 class.

            Args:
                model_name (str, optional): The name of the model to use. Defaults to "t5-base".
                training_args (Seq2SeqTrainingArguments, optional): The training arguments to use. Defaults to None.
                path_custom_logs (str, optional): The path to save custom logs. Defaults to "results".
                model_type (int, optional): The type of model to use. Defaults to 0.
                path_model_weights (str, optional): The path to the model weights. Defaults to 'results'.
                ratio (tuple, optional): The ratio for splitting the loss. Defaults to (0.5,0.5).
            """

            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.model_name = model_name
            self.model_type = model_type
            self.split_loss = False if model_type == 2 else True
            self.ratio = ratio
            self.path_custom_logs = path_custom_logs
            
            # Splits to train model on 
            self.train_split = None
            self.test_split = None
            self.dev_split = None
            self.max_length_token_input = 600
            self.max_length_token_output = 400


            # Set default training arguments
            if training_args is None:
                self.training_args = Seq2SeqTrainingArguments(
                    predict_with_generate=True,
                    evaluation_strategy="epoch",
                    num_train_epochs=5,
                    learning_rate=5e-5,
                    output_dir="./t5-base-train",
                    fp16=True

                )
            else:
                self.training_args = training_args



    def load_local_dataset(self, dataset_name: str, splits: [str], path: str) -> None:
        """
        Load the dataset from the Huggingface datasets library.

        Args:
            dataset_name (str): The name of the dataset.
            splits (List[str]): A list of split names.
            path (str): The path to the dataset files.

        Returns:
            None

        Raises:
            Exception: If there is an error loading the dataset.
        """
        try:
            train = f"{dataset_name}_{splits[0]}"
            test = f"{dataset_name}_{splits[1]}"
            dev = f"{dataset_name}_{splits[2]}"

            train_split_str, dev_split_str, test_split_str = splits

            # Load datasets
            datasets = load_dataset('json', data_files={
                train_split_str: f"{path}{train}.json",
                test_split_str: f"{path}{test}.json",
                dev_split_str: f"{path}{dev}.json"
            })

            # Access splits
            self.train_split = datasets[train_split_str]
            self.test_split = datasets[test_split_str]
            self.dev_split = datasets[dev_split_str]

            logger.success(f"Successfully loaded dataset {dataset_name} from {path}.")
        except FileNotFoundError as e:
            logger.exception(f"Error loading dataset: {e}")


    def load_and_process_dataset(self, dataset_name: str, splits: [str]):
            """
            Loads and processes a dataset.

            Args:
                dataset_name (str): The name of the dataset to load.
                splits (list[str]): The list of split names to access from the loaded dataset.

            Returns:
                None
            """
            
            dataset = load_dataset(dataset_name)

            datasets = dataset.map(
                lambda example: {'input': 'Premise: ' + example['premise'] + ' Hypothesis: ' + example['hypothesis']},
                remove_columns=['premise', 'hypothesis'],
            )

            def label_to_string(label):
                # Define your mapping from numerical label to string
                label_map = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}
                # Return the corresponding string label
                return label_map.get(label, "unknown")

            # If the dataset is the original ANLI dataset, concatenate the label and reason for generating the original rationale
            if self.model_type == 1:


                datasets = datasets.filter(
                    lambda example: example['reason'] != '' 
                )

                datasets = datasets.map(
                    lambda example : {'label2' : label_to_string(example['label'])},
                    remove_columns=['label']
                )
                datasets = datasets.map(
                    lambda example: {'target': example['label2'] + ' Explanation: ' + example['reason']},
                    remove_columns=['label2', 'reason'],
                )
            else:
                datasets = datasets.map(
                    lambda example: {'target': label_to_string(example['label'])}
                )

            processed_dataset = datasets.map(
            function=self.preprocess_data,
            batched=True)

            train_split_str, dev_split_str, test_split_str = splits

            # Access splits
            self.train_split = processed_dataset[train_split_str]
            self.test_split = processed_dataset[test_split_str]
            self.dev_split = processed_dataset[dev_split_str]




    def concat_inputs_and_targets(self, dataset: Dataset) -> Dataset:
        """
        Concatenates the inputs and targets of a dataset.

        Args:
            dataset (Dataset): The dataset to concatenate inputs and targets for.

        Returns:
            Dataset: The dataset with concatenated inputs and targets.

        Raises:
            Exception: If there is an error concatenating inputs and targets.
        """

        try:
            dataset = dataset.map(
                lambda example: {'target': example['label']  + ' Rationale: ' + example['rationale']},
                remove_columns=['label', 'rationale'],
            )
            dataset = dataset.map(
                lambda example: {'input': 'Premise: ' + example['premise'] + ' Hypothesis: ' + example['hypothesis']},
                remove_columns=['premise', 'hypothesis'],
            )
            return dataset
        except Exception as e:
            logger.exception(f"Error concatenating inputs and targets: {e}")


    def preprocess_local_data(self, inputs):
        """
        Preprocesses the data for the T5 model.

        Args:
            inputs (dict): The input data to preprocess.

        Returns:
            dict: The preprocessed input data.

        Raises:
            Exception: If there is an error preprocessing the data.
        """
        # Tokenize the inputs and targets
        model_inputs = self.tokenizer(inputs['input'], max_length=self.max_length_token_input, truncation=True,  padding='max_length')
        # print("Model Inputs: {}".format(model_inputs))
        encoded_targets = self.tokenizer([str(label) for label in inputs['target']], max_length=self.max_length_token_output, truncation=True,  padding='max_length', return_tensors="pt")
        model_inputs["labels"] = encoded_targets["input_ids"]


        decoder_input_ids = encoded_targets['input_ids']
        # Shift the `decoder_input_ids` to the right and add the start token at the beginning
        # Note: T5 uses the pad token as the start token for decoder_input_ids
        decoder_start_token_id = self.tokenizer.pad_token_id
        decoder_input_ids = torch.cat([torch.full((decoder_input_ids.shape[0], 1), decoder_start_token_id, dtype=torch.long), decoder_input_ids[:, :-1]], dim=1)
        inputs['decoder_input_ids'] = decoder_input_ids
        # print(model_inputs)
        return model_inputs



    #     return model_inputs
    def preprocess_data(self, inputs):
        # Tokenize the inputs and targets
        model_inputs = self.tokenizer(inputs['input'], max_length=self.max_length_token_input, truncation=True, padding='max_length')
        if self.model_type:

            encoded_targets = self.tokenizer([str(label) for label in inputs['target']], max_length=self.max_length_token_output, truncation=True, padding='max_length')
        

        # Assign the tokenized targets as labels
        model_inputs["labels"] = encoded_targets["input_ids"]
        
        # Prepare decoder_input_ids
        # Convert list to tensor
        decoder_input_ids = torch.tensor(encoded_targets['input_ids'], dtype=torch.long)
        
        # T5 uses the pad token as the start token for decoder_input_ids
        decoder_start_token_id = self.tokenizer.pad_token_id
        
        # Create a tensor of start tokens
        decoder_start_token_tensor = torch.full((decoder_input_ids.shape[0], 1), decoder_start_token_id, dtype=torch.long)
        
        # Concatenate start token tensor with the decoder_input_ids tensor, removing the last token to maintain size
        decoder_input_ids = torch.cat([decoder_start_token_tensor, decoder_input_ids[:, :-1]], dim=1)
        
        # Update inputs with the processed decoder_input_ids
        model_inputs['decoder_input_ids'] = decoder_input_ids.tolist() # Convert back to list if necessary
        
        return model_inputs


    def compute_exact_match(self, eval_prediction):

        predictions = eval_prediction.predictions
        labels = eval_prediction.label_ids
        preds = [self.tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]
        refs = [self.tokenizer.decode(label, skip_special_tokens=True) for label in labels]
        exact_matches = [1 if pred == ref else 0 for pred, ref in zip(preds, refs)]
        accuracy = np.mean(exact_matches)
        return {"exact_match_accuracy": accuracy}




    def prepare_training(self) -> None:
        """
        Prepare the training data for the T5 model.
        """
        try:
            self.train_split = self.concat_inputs_and_targets(self.train_split)
            self.test_split = self.concat_inputs_and_targets(self.test_split)
            self.dev_split = self.concat_inputs_and_targets(self.dev_split)
            # self.dev_split = self.dev_split.select(range(10))
            logger.success("Successfully prepared training data.")
            # logger.success(self.train_split)
        except Exception as e:
            logger.exception(f"Error preparing training data: {e}")

        # Batched processing of the data using the tokenizer defined for the T5
        try:
            self.train_split = self.train_split.map(self.preprocess_local_data, batched=True)
            # self.train_split = self.preprocess_local_data(self.train_split)
            self.test_split = self.test_split.map(self.preprocess_local_data, batched=True)
            self.dev_split = self.dev_split.map(self.preprocess_local_data, batched=True)

        except Exception as e:
            logger.exception(f"Error preprocessing data: {e}")

        # Shuffle Datasets
        self.train_split = self.train_split.shuffle(seed=42)
        self.test_split = self.test_split.shuffle(seed=42)
        self.dev_split = self.dev_split.shuffle(seed=42)



    def compute_metrics(self, eval_prediction):
        predictions = eval_prediction.predictions
        label_ids = eval_prediction.label_ids

        # Adjust decoding process to exclude -100 token IDs
        preds = []
        for pred in predictions:
            filtered_pred = [p for p in pred if p != -100]
            decoded_pred = self.tokenizer.decode(filtered_pred, skip_special_tokens=True)
            preds.append(decoded_pred)

        refs = []
        for label in label_ids:
            filtered_label = [l for l in label if l != -100]
            decoded_label = self.tokenizer.decode(filtered_label, skip_special_tokens=True)
            refs.append(decoded_label)

        label_accuracy = []
        rationale_scores = []

        precision_scores = {label: [] for label in ['Entailment', 'Neutral', 'Contradiction']}
        recall_scores = {label: [] for label in ['Entailment', 'Neutral', 'Contradiction']}
        f1_scores = {label: [] for label in ['Entailment', 'Neutral', 'Contradiction']}

        # exact_matches = [1 if pred == ref else 0 for pred, ref in zip(preds, refs)]
        if self.model_type == 1:
            for pred, ref in zip(preds, refs):

                print(f"Prediction: {pred}" )
                print(f"Reference: {ref}" )

                if " Explanation: " in pred:
                    pred_label, pred_rationale = pred.split(" Explanation: ", 1)
                    ref_label, ref_rationale = ref.split(" Explanation: ", 1)

                    label_accuracy.append(int(pred_label.strip() == ref_label.strip()))
                    rationale_scores.append(sentence_bleu([ref_rationale.strip().split()], pred_rationale.strip().split()))

                    for label in ['Entailment', 'Neutral', 'Contradiction']:
                        pred_label_binary = int(pred_label.split()[-1] == label)
                        ref_label_binary = int(ref_label.split()[-1] == label)

                        precision = precision_score([ref_label_binary], [pred_label_binary], average='binary', zero_division=0)
                        recall = recall_score([ref_label_binary], [pred_label_binary], average='binary', zero_division=0)
                        f1 = f1_score([ref_label_binary], [pred_label_binary], average='binary', zero_division=0)

                        precision_scores[label].append(precision)
                        recall_scores[label].append(recall)
                        f1_scores[label].append(f1)


        else:
            for pred, ref in zip(preds, refs):
                if " Rationale: " in pred:

                    pred_label, pred_rationale = pred.split(" Rationale: ", 1)
                    ref_label, ref_rationale = ref.split(" Rationale: ", 1)

                    label_accuracy.append(int(pred_label.strip() == ref_label.strip()))
                    rationale_scores.append(sentence_bleu([ref_rationale.strip().split()], pred_rationale.strip().split()))

                    for label in ['Entailment', 'Neutral', 'Contradiction']:
                        pred_label_binary = int(pred_label.split()[-1] == label)
                        ref_label_binary = int(ref_label.split()[-1] == label)

                        precision = precision_score([ref_label_binary], [pred_label_binary], average='binary', zero_division=0)
                        recall = recall_score([ref_label_binary], [pred_label_binary], average='binary', zero_division=0)
                        f1 = f1_score([ref_label_binary], [pred_label_binary], average='binary', zero_division=0)

                        precision_scores[label].append(precision)
                        recall_scores[label].append(recall)
                        f1_scores[label].append(f1)

        metrics = {
            "label_accuracy": np.mean(label_accuracy) if label_accuracy else 0,
            "rationale_bleu_score": np.mean(rationale_scores) if rationale_scores else 0,
            "precision": {label: np.mean(scores) for label, scores in precision_scores.items()},
            "recall": {label: np.mean(scores) for label, scores in recall_scores.items()},
            "f1_score": {label: np.mean(scores) for label, scores in f1_scores.items()},
        }

        return metrics

    


    def train(self, es: bool) -> None:
        """
        Train the T5 model.
        """
        try:
            all_metrics = {"epoch": [], "exact_match_accuracy": [], "label_accuracy": [], "rationale_bleu_score": [], "precision": [], "recall": [], "f1_score": []}

            if self.model_type == 2:
                metrics = self.compute_exact_match
            else:
                metrics = self.compute_metrics

            self.trainer = CustomTrainer(
                model=self.model,
                args=self.training_args,
                train_dataset=self.train_split,
                eval_dataset=self.dev_split,
                data_collator=default_data_collator,
                compute_metrics=metrics,
                split_loss=self.split_loss,
                ratio=self.ratio
                # callbacks=[MyCallback]
            )
            self.trainer.add_callback(CustomCallback(self.trainer, custom_logs_path=self.path_custom_logs)) 
            if es:
                self.trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=8))

            train_result = self.trainer.train()
            metrics = train_result.metrics 
            logger.success("Successfully trained T5 model.")
        except Exception as e:
            logger.exception(f"Error training T5 model: {e}")

    def test(self) -> None:
        """
        Test the T5 model.
        """
        try:
            logger.debug("Testing T5 model.")
            results = self.trainer.evaluate(self.test_split, metric_key_prefix="test")
            logger.success(f"Test results: {results}")
            self.trainer.state.save_to_json(self.path_custom_logs + "metrics.json")
        except Exception as e:
            logger.exception(f"Error testing T5 model: {e}")


    def save_model_and_tokenizer(self, path: str, model_name: str = "model") -> None:
        try:
            # Save the trained model
            model_save_path = f"{path}/{model_name}"

            os.makedirs(model_save_path, exist_ok=True)
            logger.debug(f"Path to save the model: {model_save_path}")
            self.trainer.save_model(model_save_path)

        except Exception as e:
            logger.exception(f"Couldn't save model or tokenizer: {e}")



    def run(self, dataset_name: str, splits:[], path_training_data: str, path_trained_model: str, final_model_name: str, early_stop: bool = True) -> None:
        """
        Run the T5 model.
        """
        match self.model_type:
            case 0:
                logger.debug(f"Running {self.model_name} on {dataset_name} with rationale output.")
                try:
                    self.load_local_dataset(dataset_name=dataset_name, splits=splits, path=path_training_data)
                    self.prepare_training()
                    self.train(early_stop)
                    self.test()
                    logger.success("Successfully ran T5 model.")
                    self.save_model_and_tokenizer(path=path_trained_model, model_name=final_model_name)


                except Exception as e:
                    logger.exception(f"Error running T5 model: {e}")
            case 1:
                logger.debug(f"Running {self.model_name} model on {dataset_name} with rationale output.")
                try:
                    self.load_and_process_dataset(dataset_name=dataset_name, splits=splits)
                    self.train(early_stop)
                    logger.success("Successfully ran T5 model.")
                    self.test()
                    self.save_model_and_tokenizer(path=path_trained_model, model_name=final_model_name)
                except Exception as e:
                    logger.exception(f"Error running T5 model: {e}")
            case 2:
                    
                logger.debug(f"Running {self.model_name} model on {dataset_name} without rationale output.")
                try:
                    self.load_and_process_dataset(dataset_name=dataset_name, splits=splits)
                    self.train(early_stop)
                    logger.success("Successfully ran T5 model.")
                    self.test()
                    self.save_model_and_tokenizer(path=path_trained_model, model_name=final_model_name)

                except Exception as e:
                    logger.exception(f"Error running T5 model: {e}")            