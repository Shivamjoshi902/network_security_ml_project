from network_security.exception.exception import custom_exception
import os
import sys


class network_model:
    def __init__(self, preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise custom_exception(e, sys)
    
    def predict(self, x):
        try:
            x_transform = self.preprocessor.transform(x)
            y_pred = self.model.predict(x_transform)
            return y_pred
        except Exception as e:
            raise custom_exception(e, sys)
    