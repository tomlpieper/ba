from transformers import Seq2SeqTrainer
import random
import torch
import torch.nn.functional as F
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES

class CustomTrainer(Seq2SeqTrainer):

    def __init__(self, *args, **kwargs):
        """
        Initialize the CustomTrainer object.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
                split_loss (bool): Flag indicating whether to split the loss.
                ratio (tuple): A tuple representing the ratio for splitting the loss.

        Returns:
            None
        """
        self.split_loss = kwargs.pop("split_loss", False)
        self.ratio: tuple = kwargs.pop("ratio", (0.5,0.5))
        super().__init__(*args, **kwargs)



    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Computes the loss for the given model and inputs.

        Args:
            model (torch.nn.Module): The model to compute the loss for.
            inputs (dict): The inputs to the model.
            return_outputs (bool, optional): Whether to return the outputs along with the loss. 
                Defaults to False.

        Returns:
            Union[Tuple[torch.Tensor, dict], torch.Tensor]: The computed loss. If `return_outputs` is True,
            a tuple containing the loss and the outputs is returned. Otherwise, only the loss is returned.
        """
        # raise ValueError("Inputs are: ", inputs, "Model is: ", model, "Return outputs is: ", return_outputs)
        if not self.split_loss:
            raise ValueError("Using standard loss function")
            return super().compute_loss(model, inputs, return_outputs)

        # Compute loss in a split way for the first token and the rest of the sequence
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        elif "labels" in inputs:
            print("No labels smoother are used")
            labels = inputs.pop("labels")
        else:
            labels = None
            
            
        outputs = model(**inputs) 
        
        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = unwrap_model(model)
            model_name = unwrapped_model.base_model.model._get_name() if _is_peft_model(unwrapped_model) else unwrapped_model._get_name()

            # Assuming outputs.logits shape is [batch_size, sequence_length, vocab_size]
            logits = outputs.logits

            # Split the logits and labels for the first token and the rest
            first_token_logits = logits[:, 0, :]
            rest_tokens_logits = logits[:, 1:, :]

            first_token_labels = labels[:, 0]
            rest_tokens_labels = labels[:, 1:]

            # Compute loss for the first token and the rest
            loss_fn = torch.nn.CrossEntropyLoss()
            first_token_loss = loss_fn(first_token_logits, first_token_labels)
            rest_tokens_loss = loss_fn(rest_tokens_logits.view(-1, rest_tokens_logits.size(-1)), rest_tokens_labels.view(-1))

            # Combine the two losses, giving them equal weight
            loss = 0.5 * first_token_loss + 0.5 * rest_tokens_loss
            
            raise ValueError("Computed split loss")
        else:
            # Handle case where loss is directly returned by the model
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )

            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            raise ValueError("Called custom function for loss and computed loss is: ", loss, "Returned output: ", outputs)
        return (loss, outputs) if return_outputs else loss


    def get_current_fraction(self, max_subset_size=1000):
        total_steps = len(self.get_train_dataloader()) // self.args.gradient_accumulation_steps
        current_fraction = (self.state.epoch + self.state.global_step / total_steps) / self.args.num_train_epochs

        # Calculate the maximum number of examples to include in the subset
        max_subset_size = min(max_subset_size, len(self.train_dataset))

        # Calculate the number of examples to include based on the current fraction
        subset_size = int(current_fraction * len(self.train_dataset))

        # Ensure the subset size does not exceed the maximum
        subset_size = min(subset_size, max_subset_size)

        # Ensure the subset is capped at the end (most recent examples)
        subset_start = len(self.train_dataset) - subset_size
        subset_indices = list(range(subset_start, len(self.train_dataset)))

        return current_fraction, subset_indices


    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix="eval",
        subset_fraction=None
    ):
        if subset_fraction is not None:
            total_samples = len(self.train_dataset)
            subset_size = int(subset_fraction * total_samples)

            # Randomly sample 1000 examples if the subset size is larger
            subset_size = min(subset_size, 1000)

            subset_indices = random.sample(range(total_samples), subset_size)
            eval_dataset = self.create_subset(eval_dataset, subset_indices)

        return super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix
        )

    def create_subset(self, eval_dataset, subset_indices):
        if isinstance(eval_dataset, dict):
            # Create a subset for each dataset in the dictionary
            subset_dict = {}
            for eval_dataset_name, _eval_dataset in eval_dataset.items():
                subset_dict[eval_dataset_name] = self.create_subset(_eval_dataset, subset_indices)
            return subset_dict
        else:
            # Create a subset of the dataset using the provided indices
            if hasattr(eval_dataset, "select"):
                # For datasets with a select method (e.g., datasets.Dataset)
                return eval_dataset.select(subset_indices)
            elif hasattr(eval_dataset, "iloc"):
                # For datasets with iloc method (e.g., pandas DataFrame)
                return eval_dataset.iloc[subset_indices]
            else:
                # For other types of datasets, you may need to implement your own logic
                raise ValueError("Unsupported dataset type. Implement create_subset for your dataset type.")