# from dataset_tool import DatasetTool
from openai import OpenAI
from loguru import logger
from BaseClassT5 import BaseClassT5
from transformers import T5Tokenizer, T5ForConditionalGeneration, default_data_collator, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch








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



def run_different_lrs(split_ratios: list, lr: float = 3e-4):
    for i in split_ratio:
        split_ratio_str = str(i)
        lr_str = str(lr)
        weights_path = weights_dir + "t5-small-weights/" + lr_str + "/" + split_ratio_str + "/"

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
            # gradient_accumulation_steps=2
            # remove_unused_columns=False
        )
        model = BaseClassT5(
            model_name="t5-small",
            training_args=args,
            path_custom_logs=logging_path,
            path_model_weights=weights_path,
            # original_ANLI=True,
            split_loss=True,
            ratio=j
        )

        model.run(
            dataset_name="modified_anli", 
            splits=splits[:3],
            path_training_data="v1/full_r1/",
            # path_training_data="v1/data/",
            path_trained_model=weights_path,
            final_model_name="t5-small"
        )

def run_original_anli_with_rationale(splits: [str], split_ratio: tuple, lr: float = 3e-4, use_cuda: bool = True, train_batch_size: int = 8, eval_batch_size: int = 8, logging_path: str = None, weights_path: str = None):
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
        # gradient_accumulation_steps=2
        # remove_unused_columns=False
    )
    model = BaseClassT5(
        model_name="t5-small",
        training_args=args,
        path_custom_logs=logging_path,
        path_model_weights=weights_path,
        original_ANLI=True,
        split_loss=True,
        ratio=split_ratio
    )

    model.run(
        dataset_name="anli", 
        splits=splits,
        path_training_data="v1/full_r1/",
        # path_training_data="v1/data/",
        path_trained_model=weights_path,
        final_model_name="t5-small"
    )





if __name__ == "__main__":
    # client = OpenAI()
    # tool = DatasetTool(dataset_name='anli', client=client)
    splits = ['train_r1', 'dev_r1', 'test_r1', 'train_r2', 'dev_r2', 'test_r2', 'train_r3', 'dev_r3', 'test_r3']
    # logger.success(tool.get_dataset())
    # logger.success(splits[3:6])
    # logger.success(splits[6:])

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

    logging_dir = "logs/"
    weights_dir = "results/"

    weights_path = weights_dir + "t5-small-weights-original/" + lr_str + "/" + split_ratio_str + "/"
    weights_path_labels_only = weights_dir + "t5-small-weights-labels-only/" + lr_str + "/"

    logging_path =  logging_dir + "t5-small-logs-split-loss-/" + lr_str + "_" + split_ratio_str + "/"
    logging_path_labels_only = logging_dir + "t5-small-logs-labels-only" + lr_str + "/"


    run_different_lrs(
        split_ratios=[(0.25, 0.75),(0.5,0.5), (0.75, 0.25)],
        lr=3e-4
    )






    # args = Seq2SeqTrainingArguments(
    #             predict_with_generate=True,
    #             generation_max_length=400,
    #             evaluation_strategy="steps",
    #             eval_steps=500,
    #             save_steps=1000,
    #             warmup_steps=500,
    #             per_device_train_batch_size=train_batch_size,
    #             per_device_eval_batch_size=eval_batch_size,
    #             num_train_epochs=5,
    #             learning_rate=lr,
    #             output_dir= weights_path + "outputs",
    #             fp16=use_cuda,
    #             logging_dir=weights_path + "logs",
    #             logging_steps=100,
    #             load_best_model_at_end=True,     # Load the best model when finished training (default metric is loss)
    #             metric_for_best_model='label_accuracy', # Use accuracy as the best metric unless it is baselineModel then use exact_match
    #             # gradient_accumulation_steps=2
    #          # remove_unused_columns=False
    #         )
    # model = BaseClassT5(
    #     model_name="t5-small",
    #     training_args=args,
    #     path_custom_logs=logging_path,
    #     path_model_weights=weights_path,
    #     # original_ANLI=True,
    #     split_loss=True,
    #     ratio=split_ratio
    # )

    # model.run(
    #     dataset_name="modified_anli", 
    #     splits=splits[:3],
    #     path_training_data="v1/full_r1/",
    #     # path_training_data="v1/data/",
    #     path_trained_model=weights_path,
    #     final_model_name="t5-small"
    # )
    
    
    # args_labels_only = Seq2SeqTrainingArguments(
    #         predict_with_generate=True,
    #         evaluation_strategy="steps",
    #         eval_steps=500,
    #         save_steps=1000,
    #         warmup_steps=500,
    #         per_device_train_batch_size=train_batch_size,
    #         per_device_eval_batch_size=eval_batch_size,
    #         num_train_epochs=5,
    #         learning_rate=lr,
    #         output_dir= weights_path_labels_only + "outputs",
    #         fp16=use_cuda,
    #         logging_dir=weights_path_labels_only + "logs",
    #         logging_steps=500,
    #         load_best_model_at_end=True,     # Load the best model when finished training (default metric is loss)
    #         metric_for_best_model='exact_match_accuracy',
    #         # remove_unused_columns=False
    #     )
  
    # model_labels_only = BaseClassT5(
    #     model_name="t5-small",
    #     training_args=args_labels_only,
    #     path_custom_logs=logging_path_labels_only,
    #     baseline_model=True
    # )

    # model_labels_only.run(
    #     dataset_name="anli", 
    #     splits=splits[:3],
    #     path_training_data="v1/full_r1/",
    #     # path_training_data="v1/data/",
    #     path_trained_model=logging_path_labels_only,
    #     final_model_name="t5-base-labels"
    # )
