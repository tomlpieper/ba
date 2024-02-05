from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from datasets import load_dataset
import pandas as pd
from loguru import logger
import os
import json
from tqdm import tqdm
from ratelimit import limits, sleep_and_retry



class DatasetTool:
    """
    Placeholder for a dataset tool that can be used to load a dataset and perform a rationale creation using the OpenAI API.
    """
    def __init__(self, 
        dataset_name, 
        max_length_token: int = 400, 
        temp: float = 0.2, 
        seed: int = 123,
        client = None
    ):
        self.client = client
        self.dataset_name = dataset_name
        self.max_length_token = max_length_token
        self.temp = temp
        self.seed = seed
        # self.dataset = load_dataset(self.dataset_name, split=self.split)
        self.split_dict = {}
        self.df = None




    def add_rationale_to_dataset(self, split: str, size: int) -> pd.DataFrame:

        ds = load_dataset(self.dataset_name, split=split)
        df = self.preprocess_data(ds, size=size)
        self.analyse_subset_of_data(df)
        df_unique = self.create_unique_promping_df(df)
        df_prompted = self.process_dataframe(df_unique)
        df_prompted = df_prompted[['prompt_id','rationale']]
        self.df = pd.merge(df, df_prompted, on='prompt_id', how='left')
        logger.debug('Pandas Dataframe after manipulation')
        self.analyse_subset_of_data(self.df)
        self.save_split(split, self.split_dict)

        return self.df



    @sleep_and_retry
    @limits(calls=500, period=60)  # 60 requests per minute
    def call_openai_api(self, prompt: str):
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                seed=self.seed,
                max_tokens=self.max_length_token,
                temperature=self.temp,
                messages=[
                    {"role": "system", "content": "You are helping with summarizing core semantic properties."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return None

            

    def preprocess_data(self, data, size: int) -> pd.DataFrame:

        # df = pd.DataFrame(self.dataset[part])
        df = pd.DataFrame(data)
        df = df.drop(columns=['reason']) # Drop the reason column
        df.loc[df.label == 0, 'label'] = "Entailment"
        df.loc[df.label == 1, 'label'] = "Neutral"
        df.loc[df.label == 2, 'label'] = "Contradiction"
        # Shuffle the DataFrame
        df = df.sample(frac=1, random_state=42) 
        if size != None and size < len(df):
            df = df.head(size)

        df['prompt'] = "Answer the 5 W questions about the following text with max 10 words per question:\n\n" + df['premise'] + "\n\nWho:\nWhat:\nWhen:\nWhere:\nWhy:"
        df['prompt_id'] = df.groupby('prompt').ngroup()
        logger.debug(df.shape)
        return df
    

    def process_dataframe(self, df) -> pd.DataFrame:
        """
        Process the DataFrame in parallel using a ThreadPoolExecutor calling the OpenAI API.
        args:
            df: pd.DataFrame - The DataFrame to process
        returns:
            pd.DataFrame - The processed DataFrame
        
        """
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Create a future for each API call and associate it with the DataFrame index
            future_to_index = {executor.submit(self.call_openai_api, row['prompt']): index for index, row in df.iterrows()}

            # Use tqdm to create a progress bar
            with tqdm(total=len(future_to_index), desc="Processing API calls") as pbar:
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        response = future.result()
                        rationale = response.choices[0].message.content
                        df.loc[index, 'rationale'] = rationale  # Assign the rationale directly to the DataFrame
                    except Exception as e:
                        self.logger.error(f"Error processing prompt at index '{index}': {e}")
                        df.loc[index, 'rationale'] = None  # Assign None in case of an error
                    pbar.update(1)  # Update the progress bar

        return df


    def analyse_subset_of_data(self, df) -> None:

        try:
            logger.debug(f"Number of data points: {len(df)}")
            logger.debug(f"Number of duplicates: {len(df[df.duplicated(subset=['prompt'])])}")
            logger.debug(f"Number of unique prompts: {len(df['prompt'].unique())}")
            logger.debug(f"Number of unique prompt_ids: {len(df['prompt_id'].unique())}")
            logger.debug(f"Assigned unique prompt_ids: {len(df['prompt_id'].unique()) == len(df['prompt'].unique())}")
        except Exception as e:
            logger.error(f"Error analysing data: {e}")    




    def create_unique_promping_df(self, df) -> pd.DataFrame:

        unique_prompts_df = df.drop_duplicates(subset=['prompt'])
        return unique_prompts_df



    def save_split(self, split: str, df: pd.DataFrame) -> None:
        
        self.split_dict[split] = self.df.to_dict(orient='records')
        logger.debug(f"Saved split '{split}' to split_dict")



    def write_to_dataset_csv(self, split: str, path: str = None) -> None:

        if path == None:
            path = os.getcwd()
        path = os.path.join(path, 'v1/data')

        try:
            self.df.to_csv(f"{path}/modified_{self.dataset_name}_{split}.csv", columns = ['uid','premise','hypothesis','label','rationale'], index=False)
            logger.success(f"Successfully wrote dataset to modified_{self.dataset_name}_{split}.csv")
        except Exception as e:
            logger.error(f"Error writing to csv: {e}")

    def write_dataset_to_json(self, split: str, path: str = None) -> None:

        if path == None:
            path = os.getcwd()
            path = os.path.join(path, 'v1/data')
        # logger.debug(f"Path: {path}")

        try:
            
            self.df.to_json(f"{path}/modified_{self.dataset_name}_{split}.json", orient='records')
            logger.success(f"Successfully wrote dataset to {path}/modified_{self.dataset_name}_{split}.json")
        except Exception as e:
            logger.exception(f"Error writing to json: {e}")


    def write_splits_to_json(self, path: str = None) -> None:
        if path is None:
            path = os.getcwd()
            path = os.path.join(path, 'v1/data')

        try:
            # Write the entire dictionary to a single JSON file
            with open(f"{path}/modified_{self.dataset_name}.json", 'w') as json_file:
                json.dump(self.split_dict, json_file)
            logger.success(f"Successfully wrote splits to {f'{path}/modified_{self.dataset_name}.json'}")
        except Exception as e:
            logger.error(f"Error writing splits to json: {e}")



    def get_dataset(self):
        return load_dataset(self.dataset_name)

    def get_modified_dataset(self) -> dict:
        return self.split_dict

    def get_dataset_name(self):
        return self.dataset_name

    def get_dataset_type(self):
        return self.dataset_type

    def get_split(self):
        return self.split

    def get_max_length(self):
        return self.max_length

    def get_batch_size(self):
        return self.batch_size

    def get_num_workers(self):
        return self.num_workers

    def get_dataset_size(self):
        return len(self.dataset)