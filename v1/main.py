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
    result_dir = "results/"
    logging_path =  result_dir + "t5-base-logs-split-loss" + lr_str + "/"
    logging_path_labels_only = result_dir + "t5-base-logs-labels-only" + lr_str + "/"

    args = Seq2SeqTrainingArguments(
                predict_with_generate=True,
                evaluation_strategy="steps",
                eval_steps=50,
                per_device_train_batch_size=train_batch_size,
                per_device_eval_batch_size=eval_batch_size,
                num_train_epochs=5,
                learning_rate=lr,
                output_dir= logging_path + "outputs",
                fp16=use_cuda,
                logging_dir=logging_path + "logs",
                logging_steps=50
             # remove_unused_columns=False
            )
    model = BaseClassT5(
        model_name="t5-small",
        training_args=args,
        path_custom_logs=logging_path,
        split_loss=True,
        ratio=(0.5, 0.5)
    )
    model.run(
        dataset_name="modified_anli", 
        splits=splits[:3],
        path_training_data="v1/full_r1/",
        # path_training_data="v1/data/",
        path_trained_model=logging_path,
        final_model_name="t5-small"
    )
    
    
    # args_labels_only = Seq2SeqTrainingArguments(
    #         predict_with_generate=True,
    #         evaluation_strategy="steps",
    #         eval_steps=500,
    #         per_device_train_batch_size=train_batch_size,
    #         per_device_eval_batch_size=eval_batch_size,
    #         num_train_epochs=5,
    #         learning_rate=lr,
    #         output_dir= logging_path_labels_only + "outputs",
    #         fp16=use_cuda,
    #         logging_dir=logging_path_labels_only + "logs",
    #         logging_steps=500
    #         # remove_unused_columns=False
    #     )
  
    # model_labels_only = BaseClassT5(
    #     model_name="t5-base",
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

# Run the FLAN model
    # logging_path_flan = result_dir + "flan-t5-small/"
    # logging_path_flan_labels_only = result_dir + "flan-t5-small-labels-only/"

    # args_flan = Seq2SeqTrainingArguments(
    #             predict_with_generate=True,
    #             evaluation_strategy="steps",
    #             eval_steps=500,
    #             per_device_train_batch_size=train_batch_size,
    #             per_device_eval_batch_size=eval_batch_size,
    #             num_train_epochs=5,
    #             learning_rate=5e-5,
    #             output_dir=logging_path_flan + "outputs",
    #             fp16=use_cuda,
    #             logging_dir=logging_path_flan + "logs",
    #             logging_steps=500
    #          # remove_unused_columns=False
    #         )


    # model = BaseClassT5(
    #     model_name="google/flan-t5-small",
    #     training_args=args_flan,
    #     path_custom_logs=logging_path_flan,
    #     baseline_model=False,
    #     flan=True

    # )
    # # model.run(
    # #     dataset_name="modified_anli", 
    # #     splits=splits[:3],
    # #     path_training_data="v1/full_r1/",
    # #     # path_training_data="v1/data/",
    # #     path_trained_model="v1/model3/",
    # #     final_model_name="flan-t5-small"
    # # )

    # args_flan_labels = Seq2SeqTrainingArguments(
    #             predict_with_generate=True,
    #             evaluation_strategy="steps",
    #             eval_steps=50,
    #             per_device_train_batch_size=train_batch_size,
    #             per_device_eval_batch_size=eval_batch_size,
    #             num_train_epochs=5,
    #             learning_rate=5e-5,
    #             output_dir= logging_path_flan_labels_only + "outputs",
    #             fp16=False,
    #             logging_dir=logging_path_flan_labels_only + "logs",
    #             logging_steps=50
    #          # remove_unused_columns=False
    #         )


    # model2 = BaseClassT5(
    #     model_name="google/flan-t5-small",
    #     training_args=args_flan_labels,
    #     path_custom_logs=logging_path_flan_labels_only,
    #     baseline_model=True,
    #     flan=True

    # )
    # model2.run(
    #     dataset_name="anli", 
    #     splits=splits[:3],
    #     path_training_data="v1/full_r1/",
    #     # path_training_data="v1/data/",
    #     path_trained_model="v1/model3/",
    #     final_model_name="flan-t5-small_labels_only"
    # )