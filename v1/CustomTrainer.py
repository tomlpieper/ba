from transformers import Seq2SeqTrainer
import random

class CustomTrainer(Seq2SeqTrainer):

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