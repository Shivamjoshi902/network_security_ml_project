from network_security.entity.config_entity import DataTransformationConfig
from network_security.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from network_security.exception.exception import custom_exception
from network_security.utils.main_utils import save_numpy_array_data, save_object
from network_security.constant.training_pipeline import TARGET_COLUMN, DATA_TRANSFORMATION_IMPUTER_PARAMS
from network_security.logging.logger import logging
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import sys
import os

try:
    pass
except Exception as e:
    raise custom_exception(e, sys)

class data_transformation:
    def __init__(self, data_transformation_config : DataTransformationConfig, data_validation_artifact : DataValidationArtifact):
        try:
            self.data_transformation_config: DataTransformationConfig = data_transformation_config
            self.data_validation_artifact: DataValidationArtifact = data_validation_artifact
        except Exception as e:
            raise custom_exception(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise custom_exception(e, sys)


    def get_preprocessor_object(cls)->Pipeline:
        try:
            imputer:KNNImputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            logging.info(
                f"Initialise KNNImputer with {DATA_TRANSFORMATION_IMPUTER_PARAMS}"
            )
            processor:Pipeline = Pipeline([
                ('imputer', imputer)
            ])
            return processor
        
        except Exception as e:
            raise custom_exception(e, sys)


    def initiate_data_transformation(self)->DataTransformationArtifact:
        logging.info('reading train and test files')

        train_df = self.read_data(self.data_validation_artifact.valid_train_file_path)
        test_df = self.read_data(self.data_validation_artifact.valid_test_file_path)

        logging.info('seperating target and input columns for train and test data')
        
        input_features_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
        target_feature_train_df = train_df[TARGET_COLUMN]
        target_feature_train_df = target_feature_train_df.replace(-1, 0)

        input_features_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
        target_feature_test_df = test_df[TARGET_COLUMN]
        target_feature_test_df = target_feature_test_df.replace(-1, 0)

        logging.info('creating preprocessor object')
        preprocessor: Pipeline = self.get_preprocessor_object()
        preprocessor_obj = preprocessor.fit(input_features_train_df)
        transformed_input_features_train_df = preprocessor_obj.transform(input_features_train_df)
        transformed_input_features_test_df = preprocessor_obj.transform(input_features_test_df)

        train_arr = np.c_[transformed_input_features_train_df, np.array(target_feature_train_df)]
        test_arr = np.c_[transformed_input_features_test_df, np.array(target_feature_test_df)]

        logging.info('storing transformed train and test array and preprocessor object')
        save_numpy_array_data( self.data_transformation_config.transformed_train_file_path, array=train_arr, )
        save_numpy_array_data( self.data_transformation_config.transformed_test_file_path,array=test_arr,)
        save_object( self.data_transformation_config.transformed_object_file_path, preprocessor_obj)

        save_object( "final_model/preprocessor.pkl", preprocessor_obj)

        data_transformation_artifact = DataTransformationArtifact(
            transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
            transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
            transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
        )
        return data_transformation_artifact

