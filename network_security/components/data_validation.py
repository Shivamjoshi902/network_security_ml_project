from network_security.entity.config_entity import DataValidationConfig
from network_security.entity.artifact_entity import data_ingetion_artifacts
from network_security.entity.artifact_entity import DataValidationArtifact
from network_security.constant.training_pipeline import SCHEMA_FILE_PATH
from network_security.exception.exception import custom_exception
from network_security.logging.logger import logging
from network_security.utils.main_utils import read_yaml_file, write_yaml_file
from scipy.stats import ks_2samp
import pandas as pd
import sys
import os

try:
    pass
except Exception as e:
    raise custom_exception(e, sys)

class data_validation:
    def __init__(self, data_validation_config :DataValidationConfig, 
                 data_ingetion_artifacts: data_ingetion_artifacts):
        try:
            self.data_validation_config = data_validation_config
            self.data_ingetion_artifacts = data_ingetion_artifacts
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        
        except Exception as e:
            raise custom_exception(e, sys)
    
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise custom_exception(e,sys)
    
    def validate_no_of_columns(self, df:pd.DataFrame)->bool:
        try:
            no_of_columns = len(self.schema_config['columns'])
            logging.info(f"Required number of columns:{no_of_columns}")
            logging.info(f"Data frame has columns:{len(df.columns)}")
            if no_of_columns == len(df.columns):
                return True
            return False
        
        except Exception as e:
            raise custom_exception(e, sys)

    def detect_data_drift(self, base_df, current_df, threshold = 0.05):
        try:
            status = True
            report = {}
            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                data_drift = ks_2samp(d1, d2)
                if threshold <= data_drift.pvalue:
                    is_found = False
                else:
                    is_found = True
                    status = False
                
                report.update({
                    column:{
                        'pvalue' : float(data_drift.pvalue),
                        'drift_status': is_found
                    }
                })
            drift_report_file_path = self.data_validation_config.drift_report_file_path

            #Create directory
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path,exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path,content=report)
        
        except Exception as e:
            raise custom_exception(e, sys)

    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            train_file_path = self.data_ingetion_artifacts.train_file_path
            test_file_path = self.data_ingetion_artifacts.test_file_path

            logging.info('reading train and test df')
            test_df = self.read_data(test_file_path)
            train_df = self.read_data(train_file_path)


            logging.info('starting validation of no of columns in train and test df')

            train_df_validation_status=self.validate_no_of_columns(df = train_df)
            if not train_df_validation_status:
                error_message=f"Train dataframe does not contain all columns.\n"
            
            test_df_validation_status = self.validate_no_of_columns(df = test_df)
            if not test_df_validation_status:
                error_message=f"Test dataframe does not contain all columns.\n"   


            logging.info('starting validation to find any drift in data')

            status=self.detect_data_drift(base_df=train_df,current_df=test_df)
            dir_path=os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path,exist_ok=True)

            train_df.to_csv(
                self.data_validation_config.valid_train_file_path, index=False, header=True
            )
            test_df.to_csv(
                self.data_validation_config.valid_test_file_path, index=False, header=True
            )

            data_validation_artifact = DataValidationArtifact(
                validation_status = (train_df_validation_status and test_df_validation_status),
                valid_train_file_path = self.data_ingetion_artifacts.train_file_path,
                valid_test_file_path = self.data_ingetion_artifacts.test_file_path,
                invalid_train_file_path = None,
                invalid_test_file_path = None,
                drift_report_file_path = self.data_validation_config.drift_report_file_path,
            )
            return data_validation_artifact
            
        except Exception as e:
            raise custom_exception(e, sys)
