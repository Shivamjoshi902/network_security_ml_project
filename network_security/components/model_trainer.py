from network_security.entity.config_entity import ModelTrainerConfig
from network_security.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact
from network_security.exception.exception import custom_exception
from network_security.utils.ml_utils.model.estimator import network_model
from network_security.utils.main_utils import save_object, load_numpy_arr_data, evaluate_model, load_pkl_object
from network_security.logging.logger import logging
from network_security.utils.ml_utils.metrix.classification_metrix import get_classification_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
import pandas as pd
import numpy as np
import sys
import os

try:
    pass
except Exception as e:
    raise custom_exception(e, sys)

class model_trainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise custom_exception(e, sys)
    
    def train_model(self, x_train, y_train, x_test, y_test)->ModelTrainerArtifact:
        models = {
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "Logistic Regression": LogisticRegression(verbose=1),
                "AdaBoost": AdaBoostClassifier(),
            }
        params={
            "Decision Tree": {
                'criterion':['gini', 'entropy', 'log_loss'],
                # 'splitter':['best','random'],
                # 'max_features':['sqrt','log2'],
            },
            "Random Forest":{
                # 'criterion':['gini', 'entropy', 'log_loss'],
                
                # 'max_features':['sqrt','log2',None],
                'n_estimators': [8,16,32,128,256]
            },
            "Gradient Boosting":{
                # 'loss':['log_loss', 'exponential'],
                'learning_rate':[.1,.01,.05,.001],
                'subsample':[0.6,0.7,0.75,0.85,0.9],
                # 'criterion':['squared_error', 'friedman_mse'],
                # 'max_features':['auto','sqrt','log2'],
                'n_estimators': [8,16,32,64,128,256]
            },
            "Logistic Regression":{},
            "AdaBoost":{
                'learning_rate':[.1,.01,.001],
                'n_estimators': [8,16,32,64,128,256]
            }
            
        }

        report:dict = evaluate_model(X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test, models=models, params=params)

        best_model_score = max(sorted(report.values()))
        best_model_name = list(report.keys())[
            list(report.values()).index(best_model_score)
        ]

        best_model = models[best_model_name]

        y_train_pred = best_model.predict(x_train)
        train_classification_articact = get_classification_score(y_true=y_train, y_pred=y_train_pred)

        y_test_pred = best_model.predict(x_test)
        test_classification_articact = get_classification_score(y_true=y_test, y_pred=y_test_pred)

        preprocessor = load_pkl_object(self.data_transformation_artifact.transformed_object_file_path)

        network_model_obj = network_model(preprocessor=preprocessor, model=best_model)

        path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(path, exist_ok=True)

        save_object(self.model_trainer_config.trained_model_file_path, network_model_obj)
        save_object('final_model/model.pkl', best_model)

        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=train_classification_articact,
            test_metric_artifact=test_classification_articact
        )
        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact




    def initiate_model_training(self):
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            logging.info('reading test and train array')
            train_arr = load_numpy_arr_data(train_file_path)
            test_arr = load_numpy_arr_data(test_file_path)

            x_train, y_train, x_test, y_test = [
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            ]

            model_trainer_artifact = self.train_model(x_train, y_train, x_test, y_test)     
            return model_trainer_artifact
        
        except Exception as e:
            raise custom_exception(e, sys)
            
