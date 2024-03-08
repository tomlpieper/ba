# from dataset_tool import DatasetTool
# from openai import OpenAI
from loguru import logger
from BaseClassT5 import BaseClassT5
from transformers import T5Tokenizer, T5ForConditionalGeneration, default_data_collator, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch
import mlflow
import os

from wrapped_runs import run_modified_anli_with_rationale, run_original_anli_without_rationale, run_original_anli_with_rationale

torch.distributed.init_process_group('NCCL')
local_rank = int(os.environ['LOCAL_RANK'])
# torch.cuda.set_device(local_rank)
# device = torch.device("cuda", local_rank)
# cuda_visible = os.environ['CUDA_VISIBLE_DEVICES']
# print(f"Device: {device}")
# print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")c
print(f"NCCL found: {torch.distributed.is_nccl_available()}")
# print(f"Local Rank: {local_rank}")

def create_modified_dataset(
    splits: list, 
    amount_training_examples: int = None, 
    path: str = None
    ) -> None:
    """
    Create a modified dataset with rationale added to the original dataset.
    """
    os.makedirs(path, exist_ok=True)
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
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")
    use_cuda = torch.cuda.is_available()
    print(f"Using GPU: {use_cuda}")
    # train_batch_size = 8 if use_cuda else 1
    # eval_batch_size = 8 if use_cuda else 1

    # Batch size for T5-small on 2x RTXA6000 
    train_batch_size = 64 if use_cuda else 1
    eval_batch_size = 64 if use_cuda else 1

    # Batch size for T5-base on 2x RTXA6000 
    # train_batch_size = 24 if use_cuda else 1
    # eval_batch_size = 24 if use_cuda else 1


    # create_modified_dataset(splits[:3], amount_training_examples=100, path='/Users/tompieper/code_3/v2/full_r1/')
    # create_modified_dataset(splits[3:6], amount_training_examples=100000, path='v1/full_r2/')
    # create_modified_dataset(splits[6:], amount_training_examples=1000000, path='v1/full_r3/')
    # create_modified_dataset(['test_r1'], amount_training_examples=100, path='data/')
    # lr = 3e-4
    lr = 0.0024
    lr_str = str(lr)
    # split_ratio = (0.25, 0.75)
    split_ratio = (0.75, 0.25)
    split_ratio_str = str(split_ratio)


    # run_original_anli_without_rationale(
    #     splits=splits[:3],
    #     lr=lr,
    #     train_batch_size=32,
    #     eval_batch_size=32,
    #     eval_steps=132,
    #     base_path="/netscratch/tpieper/",
    #     data_path="/netscratch/tpieper/v1/full_r1/",
    #     model_name="t5-small"
    # )

    # lr = lr*
    # Learning rate increased by number



    # lrs = [6e-4, 0.0012, 0.0024]
    # for i in lrs:
    #     run_modified_anli_with_rationale(
    #         splits=splits[:3], 
    #         split_ratio=(0.5,0.5), 
    #         lr=i, 
    #         epochs=5,
    #         use_cuda=True, 
    #         train_batch_size=32, 
    #         eval_batch_size=32,
    #         eval_steps=132, 
    #         base_path="/netscratch/tpieper/",
    #         data_path="/netscratch/tpieper/v1/full_r1/",
    #         model_name="t5-small"
    #     )

    # Run different loss ratios on cluster for mR1 and T5-Small
    # for i in [
    #     (0.25, 0.75),
    #     (0.5,0.5), 
    #     (0.75, 0.25)
    #     ]: 
    #     for j in [
    #         6e-4, 
    #         0.0012, 
    #         0.0024
    #         ]:
    #         run_modified_anli_with_rationale(
    #             splits=splits[:3], 
    #             split_ratio=i, 
    #             lr=j, 
    #             epochs=5,
    #             use_cuda=True, 
    #             train_batch_size=32, 
    #             eval_batch_size=32,
    #             eval_steps=132, 
    #             base_path="/netscratch/tpieper/v2",
    #             data_path="/netscratch/tpieper/v1/full_r1/",
    #             model_name="t5-small"
    #         )

    # run_original_anli_with_rationale(
    #     splits=splits[:3],
    #     split_ratio=(0.25,0.75),
    #     lr=0.0006,
    #     epochs=5,
    #     use_cuda=False,
    #     train_batch_size=32,
    #     eval_batch_size=32,
    #     eval_steps=20,
    #     base_path="/netscratch/tpieper/v2.1/",
    #     data_path="/netscratch/tpieper/v1/full_r1/",
    #     model_name="t5-small"
#    )


    # run_modified_anli_with_rationale(
    #      splits=splits[:3], 
    #             split_ratio=(0.5,0.5),
    #             lr=0.0006,
    #             epochs=5,
    #             use_cuda=True, 
    #             train_batch_size=32, 
    #             eval_batch_size=32,
    #             eval_steps=132, 
    #             base_path="/netscratch/tpieper/v3/",
    #             data_path="/netscratch/tpieper/v2/full_r1/",
    #             model_name="t5-small"
    # )

    # Run original ANLI without rationale on V2
    for i in [6e-4, 0.0012, 0.0024]:
        run_original_anli_without_rationale(
            splits=splits[:3],
            lr=i,
            train_batch_size=32,
            eval_batch_size=32,
            eval_steps=132,
            base_path="/netscratch/tpieper/v3/baseline/",
            data_path="/netscratch/tpieper/v2/full_r1/",
            model_name="t5-small"
        )

    # Run V2 Dataset for different split ratios and learning rates
    for i in [
        (0.25, 0.75),
        (0.5,0.5),
        (0.75, 0.25)
        ]:
        for j in [
            6e-4,
            0.0012,
            0.0024
            ]:
            run_modified_anli_with_rationale(
                splits=splits[:3],
                split_ratio=i,
                lr=j,
                epochs=5,
                use_cuda=True,
                train_batch_size=32,
                eval_batch_size=32,
                eval_steps=132,
                base_path="/netscratch/tpieper/v3/",
                data_path="/netscratch/tpieper/v2/full_r1/",
                model_name="t5-small"
            )
    # logger.debug("Finished running modified ANLI with rationale on V2.")



model_types = {
    0: "modified_anli_with_rationale", 
    1: "original_anli_with_rationale", 
    2: "orignal_anli_without_rationale"}
