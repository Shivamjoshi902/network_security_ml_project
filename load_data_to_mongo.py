import os
import sys
import json
from dotenv import load_dotenv

load_dotenv()
mongo_db_url = os.getenv('mongo_db_url')

import certifi
ca=certifi.where()

import numpy as np
import pandas as pd
import pymongo
from network_security.exception.exception import custom_exception
from network_security.logging.logger import logging

class network_data_extract:
    def __init__(self):
        pass

    def csv_to_json_converter(self, file_path):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            record = data.to_dict(orient="records")
            return record

        except Exception as e:
            raise custom_exception(e, sys)
    
    def insert_data_to_mongo(self, database, collection, records):
        try:
            self.collection = collection
            self.database = database
            self.records = records

            self.mongo_client=pymongo.MongoClient(mongo_db_url)
            self.database = self.mongo_client[self.database]
            
            self.collection=self.database[self.collection]
            self.collection.insert_many(self.records)
            return(len(self.records))
        
        except Exception as e:
            raise custom_exception(e, sys)

if __name__ == '__main__':
    file_path = 'Network_Data\phisingData.csv'
    database = 'network_db'
    collection = 'network_collection'

    obj = network_data_extract()
    records = obj.csv_to_json_converter(file_path)
    print(records)
    length_of_inserted_data = obj.insert_data_to_mongo(database=database, collection=collection, records=records)
    print(length_of_inserted_data)

