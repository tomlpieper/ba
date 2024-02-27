import warnings
warnings.simplefilter("ignore", UserWarning)
from datasets import load_dataset, Dataset
import pandas as pd
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration, default_data_collator, Seq2SeqTrainingArguments, Seq2SeqTrainer, TrainerCallback, AutoModelForSeq2SeqLM, AutoTokenizer
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
    
    def __init__(self, model_name: str = "t5-base", training_args: Seq2SeqTrainingArguments = None, path_custom_logs: str = "results", baseline_model: bool = False, path_model_weights: str = 'results', flan: bool = False, split_loss: bool = False, ratio: tuple = (0.5,0.5) ) -> None:
            """
            Initializes an instance of the BaseClassT5.

            Args:
                model_name (str): The name of the T5 model to be used. Defaults to "t5-base".
                training_args (Seq2SeqTrainingArguments): The training arguments for the model. Defaults to None.
            """

            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.model_name = model_name
            self.split_loss = split_loss
            self.ratio = ratio
            self.baseline_model = baseline_model
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
                    # per_device_train_batch_size=1,
                    per_device_eval_batch_size=1,
                    num_train_epochs=5,
                    learning_rate=5e-5,
                    output_dir="./t5-base-train",
                    fp16=True
                 # remove_unused_columns=False
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

            # Load datasets
            datasets = load_dataset('json', data_files={
                "train_r1": f"{path}{train}.json",
                "test_r1": f"{path}{test}.json",
                "dev_r1": f"{path}{dev}.json"
            })

            # Access splits
            self.train_split = datasets['train_r1']
            self.test_split = datasets['test_r1']
            self.dev_split = datasets['dev_r1']

            logger.success(f"Successfully loaded dataset {dataset_name} from {path}.")
        except FileNotFoundError as e:
            logger.exception(f"Error loading dataset: {e}")


    def load_and_process_dataset(self, dataset_name: str, splits: [str]):
        dataset = load_dataset(dataset_name)
        datasets = dataset.map(
            lambda example: {'input': 'Premise: ' + example['premise'] + ' Hypothesis: ' + example['hypothesis']},
            remove_columns=['premise', 'hypothesis'],
        )
        processed_dataset = datasets.map(
        function=self.preprocess_data,
        batched=True)
        self.train_split = processed_dataset['train_r1']
        self.test_split = processed_dataset['test_r1']
        self.dev_split = processed_dataset['dev_r1']


    def concat_inputs_and_targets(self, dataset: Dataset) -> Dataset:
        """
        Concatenate the inputs and outputs in a way that the T5 learns what to predict.
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


    def preprocess_data(self,inputs):
        # print("Inputs: {}".format(inputs))
        model_inputs = self.tokenizer(inputs['input'], max_length=self.max_length_token_input, truncation=True,  padding='max_length')
        # print("Model Inputs: {}".format(model_inputs))
        labels = self.tokenizer([str(label) for label in inputs['label']], max_length=self.max_length_token_output, truncation=True,  padding='max_length')
        model_inputs["labels"] = labels["input_ids"]
        # print(f"Model Inputs: {model_inputs}")
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
        preds = [self.tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]
        refs = [self.tokenizer.decode(label, skip_special_tokens=True) for label in label_ids]

        label_accuracy = []
        rationale_scores = []

        precision_scores = {label: [] for label in ['Entailment', 'Neutral', 'Contradiction']}
        recall_scores = {label: [] for label in ['Entailment', 'Neutral', 'Contradiction']}
        f1_scores = {label: [] for label in ['Entailment', 'Neutral', 'Contradiction']}

        exact_matches = [1 if pred == ref else 0 for pred, ref in zip(preds, refs)]

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
    


    def train(self) -> None:
        """
        Train the T5 model.
        """
        try:
            all_metrics = {"epoch": [], "exact_match_accuracy": [], "label_accuracy": [], "rationale_bleu_score": [], "precision": [], "recall": [], "f1_score": []}

            if self.baseline_model:
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

            train_result = self.trainer.train()
            metrics = train_result.metrics 
            logger.success("Successfully trained T5 model.")
        except Exception as e:
            logger.exception(f"Error training T5 model: {e}")




    def save_model_and_tokenizer(self, path: str, model_name: str = "model") -> None:
        try:
            # Save the trained model
            model_save_path = f"{path}/{model_name}"

            os.makedirs(model_save_path, exist_ok=True)
            logger.debug(f"Path to save the model: {model_save_path}")
            self.trainer.save_model(model_save_path)

        except Exception as e:
            logger.exception(f"Couldn't save model or tokenizer: {e}")



    def run(self, dataset_name: str, splits:[], path_training_data: str, path_trained_model: str, final_model_name: str) -> None:
        """
        Run the T5 model.
        """
        if not self.baseline_model:
            logger.debug(f"Running {self.model_name} on {dataset_name} with rationale output.")
            try:
                self.load_local_dataset(dataset_name=dataset_name, splits=splits, path=path_training_data)
                self.prepare_training()
                self.train()
                logger.success("Successfully ran T5 model.")
                self.save_model_and_tokenizer(path=self.path_custom_logs, model_name=final_model_name)


            except Exception as e:
                logger.exception(f"Error running T5 model: {e}")


        else:
            logger.debug(f"Running {self.model_name} model on {dataset_name} without rationale output.")
            try:
                self.load_and_process_dataset(dataset_name=dataset_name, splits=splits)
                logger.debug(self.train_split[0])
                self.train()
                logger.success("Successfully ran T5 model.")
                self.save_model_and_tokenizer(path=path_trained_model, model_name=final_model_name)

            except Exception as e:
                logger.exception(f"Error running T5 model: {e}")


