from transformers import TrainerCallback, TrainerControl
from copy import deepcopy
import os
from loguru import logger
import json
import torch
import numpy as np




class CustomCallback(TrainerCallback):
    "A callback that prints a message at the beginning of training"

    def __init__(self, trainer, custom_logs_path: str = "results" ) -> None:
        super().__init__()
        self._trainer = trainer
        self._evaluated = False
        self._finished = False
        self.custom_logs_path = custom_logs_path
        os.makedirs(custom_logs_path, exist_ok=True)

    def on_train_begin(self, args, state, control, **kwargs):
        logger.info("Starting training")

    def on_train_end(self, args, state, control, **kwargs):
        logger.info("Finished training")
        self._finished = True
        # dc = self.evaluate_on_training_data(state, control, train_and_eval=True)
        # if dc:
        #     return dc

    def on_init_end(self, args,  state, control, **kwargs):
        # ("Finished init of trainer")
        os.makedirs(self.custom_logs_path, exist_ok=True)


    def on_log(self, args, state, control, **kwargs):
        pass


    def on_save(self, args, state, control, **kwargs):
        pass



    def on_epoch_end(self, args, state, control, **kwargs):
        pass

    def on_evaluate(self, args, state, control, **kwargs):
        state.save_to_json(self.custom_logs_path + "eval_metrics.json")
         # pass
        # if not self._finished:
        #     dc = self.evaluate_on_training_data(state, control)
        #     if dc:
        #         return dc
       

    def evaluate_on_training_data(self, state, control, train_and_eval:bool = False, **kwargs):

        print("\n")
        # logger.info("Evaluating model on Training dat")

        if not self._evaluated:
            self._evaluated = True
            control_copy = deepcopy(control)
            if train_and_eval:
                self._trainer.evaluate(metric_key_prefix="eval")
            frac, size = self._trainer.get_current_fraction()
            logger.debug(f"Subset of training data is {frac/2} of dataset, i.e. {len(size)} examples.")
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train", subset_fraction = frac)
            state.save_to_json(self.custom_logs_path + "eval_metrics.json")
            return control_copy
        else:
            self._evaluated = False
        