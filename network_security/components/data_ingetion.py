import os
from datetime import datetime
from network_security.exception.exception import custom_exception
from network_security.logging.logger import logging

from network_security.entity.config_entity import DataIngestionConfig
from network_security.entity.artifact_entity import data_ingetion_artifacts
import sys
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import pymongo
from typing import List
from sklearn.model_selection import train_test_split
load_dotenv()

mongo_db_url = os.getenv('mongo_db_url')

class data_ingestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
            logging.info('data_ingestion object created')
        except Exception as e:
            raise custom_exception(e, sys)
    
    def export_collection_as_df(self):
        try:
            database = self.data_ingestion_config.database_name
            collection = self.data_ingestion_config.collection_name
            self.mongo_client = pymongo.MongoClient(mongo_db_url)
            collection = self.mongo_client[database][collection]

            df = pd.DataFrame(list(collection.find()))

            if '_id' in df.columns.tolist():
                df.drop(columns=['_id'], axis=1, inplace=True)

            df.replace({"na":np.nan},inplace=True)
            logging.info('imported data from mongo and converted to df')
            return df
        
        except Exception as e:
            raise custom_exception(e, sys)

    def save_df_in_feature_store(self, df:pd.DataFrame):
        try:
            feature_store_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_path)
            os.makedirs(dir_path, exist_ok=True)
            df.to_csv(feature_store_path, index=False, header=True)
            logging.info('df saved as raw file in feature store path')

        except Exception as e:
            raise custom_exception(e, sys)

    def split_data_as_test_train(self, df: pd.DataFrame):
        try:
            logging.info(f"splitting df into train and test.")
            train_data, test_data = train_test_split( df, test_size=self.data_ingestion_config.train_test_split_ratio )

            dir_path_test = os.path.dirname(self.data_ingestion_config.testing_file_path)
            dir_path_train = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path_test, exist_ok=True)
            os.makedirs(dir_path_train, exist_ok=True)

            logging.info(f"Exporting train and test file.")
            train_data.to_csv(self.data_ingestion_config.training_file_path, index = False, header = True)
            test_data.to_csv(self.data_ingestion_config.testing_file_path, index = False, header = True)
            logging.info(f"train and test file Exported.")

        except Exception as e:
            raise custom_exception(e, sys)

        
    def initiate_data_ingestion(self):
        try:
            df = self.export_collection_as_df()
            self.save_df_in_feature_store(df=df)
            self.split_data_as_test_train(df=df)

            data_ingetion_artifact = data_ingetion_artifacts(
                train_file_path=self.data_ingestion_config.training_file_path, 
                test_file_path=self.data_ingestion_config.testing_file_path
            )
            return data_ingetion_artifact
        except Exception as e:
            raise custom_exception(e, sys)
        




